import numpy


def ransac(data, s, t, d):
    """fit model parameters to data using the RANSAC algorithm
This implementation is meant to fit affine transforms which requires 3 corresponding points.
This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

{{{
Given:
    data - a set of observed data points
    s - the minimum number of data values required to fit the model (minimal set)
    number_of_iterations - the maximum number of iterations allowed in the algorithm
    t - how har away from the model is acceptable to an inliear
    d - the number of close data values required to assert that a model fits well to data
Return:
    best_fit - model parameters which best fit the data (or nil if no good model is found)
iterations = 0
best_fit = nil
best_err = something really large
while iterations < number_of_iterations {
    candidate_inliears = n randomly selected values from data
    maybemodel = model parameters fitted to candidate_inliears
    alsoinliers = empty set
    for every point in data not in candidate_inliears {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in candidate_inliears and alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < best_err {
            best_fit = bettermodel
            best_err = thiserr
        }
    }
    increment iterations
}
return best_fit
}}}
"""
    iterations = 0
    best_fit = None
    best_err = numpy.inf
    best_inlier_idxs = None
    p = 0.99 # probability of success, i.e prob of one of the candidate inliers will be a correct set of points
    e = 0.7 # proportion of outliers outliers
    number_of_iterations = numpy.log(1-p)/numpy.log(1-(1-e)**s)
    print("# of interations:", number_of_iterations)
    while iterations < int(number_of_iterations+1):
        # randomly select s points (or point pairs) to form a sample
        candidate_inliears_ids, test_idxs = random_partition(s, data.shape[0])
        candidate_inliears = data[candidate_inliears_ids, :]
        test_points = data[test_idxs]

        # instantiate the model (homography)
        maybemodel = fit_homography(candidate_inliears, s)
        test_err = get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t]  # select indices of rows with accepted points
        alsoinliers = data[also_idxs, :]
        print ('test_err.min()', test_err.min())
        print ('test_err.max()', test_err.max())
        print ('numpy.mean(test_err)', numpy.mean(test_err))
        print ('iteration %d:len(alsoinliers) = %d' % (
            iterations, len(alsoinliers)))
        if len(alsoinliers) > d:
            betterdata = numpy.concatenate((candidate_inliears, alsoinliers))
            bettermodel = fit_homography(betterdata, s)
            better_errs = get_error(betterdata, bettermodel)
            thiserr = numpy.mean(better_errs)
            if thiserr < best_err:
                best_fit = bettermodel
                best_err = thiserr
                best_inlier_idxs = numpy.concatenate((candidate_inliears_ids, also_idxs))
        iterations += 1
    if best_fit is None:
        raise ValueError("did not meet fit acceptance criteria")
    else:
        return best_fit, {'inliers': best_inlier_idxs}

def normalize(points):
    """ Normalize a collection of points in
        homogeneous coordinates so that last row = 1. """

    for row in points:
        row /= points[-1]
    return points


def fit_homography(data, s):
    """ Fit homography to four selected correspondences. """
    # transpose to fit H
    data = data.T

    # from points
    fp = data[:3, :s]

    # target points
    tp = data[3:, :s]

    # fit homography and return
    return Haffine_from_points(fp, tp)

def get_error(data, H):
    """ Apply homography to all correspondences,
        return error for each transformed point. """

    data = data.T

    # from points
    fp = data[:3]
    # target points
    tp = data[3:]

    # transform fp
    fp_transformed = numpy.dot(H, fp)

    # normalize hom. coordinates
    fp_transformed = normalize(fp_transformed)

    # return error per point
    return numpy.sqrt(numpy.sum((tp - fp_transformed) ** 2, axis=0))


def Haffine_from_points(fp, tp):
    """ Find H, affine transformation, such that
        tp is affine transf of fp. """

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition points
    # --from points--
    m = numpy.mean(fp[:2], axis=1)
    maxstd = max(numpy.std(fp[:2], axis=1)) + 1e-9
    C1 = numpy.diag([1 / maxstd, 1 / maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = numpy.dot(C1, fp)

    # --to points--
    m = numpy.mean(tp[:2], axis=1)
    C2 = C1.copy()  # must use same scaling for both point sets
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = numpy.dot(C2, tp)

    # conditioned points have mean zero, so translation is zero
    A = numpy.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = numpy.linalg.svd(A.T)

    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = numpy.concatenate((numpy.dot(C, numpy.linalg.pinv(B)), numpy.zeros((2, 1))), axis=1)
    H = numpy.vstack((tmp2, [0, 0, 1]))

    # decondition
    H = numpy.dot(numpy.linalg.inv(C2), numpy.dot(H, C1))

    return H / H[2, 2]

def random_partition(s, number_of_points):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = numpy.arange(number_of_points)
    numpy.random.shuffle(all_idxs)

    # first get the minimum set necessary data points, 4 in the case of homography
    minimum_set_datapoints = all_idxs[:s]

    # select the other points
    idxs2 = all_idxs[s:]
    return minimum_set_datapoints, idxs2


def H_from_ransac(fp, tp, match_theshold=10):
    """ Robust estimation of homography H from point
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).

        input: fp,tp (3*n arrays) points in hom. coordinates. """

    # group corresponding points
    data = numpy.concatenate((fp, tp), axis=1)

    # compute H and return
    H, ransac_data = ransac(data, 3, match_theshold, 10)
    return H, ransac_data['inliers']


def convert_to_homogeneous_coord(points):
    """ Convert a set of points (dim*n array) to
        homogeneous coordinates. """
    return numpy.concatenate((points, numpy.ones((points.shape[0],1))), axis=1)

