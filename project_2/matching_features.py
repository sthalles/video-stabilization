import numpy

def l2_norm(data):
    return numpy.sqrt(numpy.sum(numpy.abs(data) ** 2))


def match_features(img_descriptors1, img_descriptors2):
    # The first thing we need to do is to normlize the feature vectors using the Frobenius norm
    norm_desc1 = numpy.array([des / l2_norm(des) for des in img_descriptors1])
    norm_desc2 = numpy.array([des / l2_norm(des) for des in img_descriptors2])

    # we are going to use the ratio of distance from one feature in image 1
    # to the two closest matching features in image 2
    distance_ratio = 0.6

    match_scores = numpy.zeros((img_descriptors1.shape[0]), dtype=numpy.int)

    # loop through each descriptor
    norm_desc2_t = norm_desc2.T  # precompute matrix transpose so that we do not need to repeat this operation
    for i in range(norm_desc1.shape[0]):
        # multiply each feature point from image 1 by the transpose of the 2nd image feature points
        dot_prods = numpy.dot(norm_desc1[i, :], norm_desc2_t)

        dot_prods = 0.9999 * dot_prods

        # inverse cosine
        dot_prods_arccos = numpy.arccos(dot_prods)

        # sort, return index for features in second image
        indx = numpy.argsort(dot_prods_arccos)

        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if dot_prods_arccos[indx[0]] < distance_ratio * dot_prods_arccos[indx[1]]:
            match_scores[i] = int(indx[0])

    return match_scores


def match_symmetric(desc1, desc2):
    """ Two-sided symmetric version of match(). """

    matches_12 = match_features(desc1, desc2)
    matches_21 = match_features(desc2, desc1)

    # take only the non zero matches
    ndx_12 = matches_12.nonzero()[0]

    # discard matches that are not symmetric between the two frames
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12