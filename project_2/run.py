import cv2
import numpy
from ransac import H_from_ransac, convert_to_homogeneous_coord
#from matplotlib import pyplot as plt
import math
from matching_features import match_symmetric

def convert_affine_to_scale_rotation_trainlation_transform(tform):

    #Extract scale and rotation part sub-matrix.
    H = tform
    R = H[0:2,0:2]
    #print("R", R)
    # Compute theta from mean of two possible arctangents
    theta = numpy.mean([math.atan2(R[1,0],R[0,0]), math.atan2(-R[0,1],R[1,1])]);
    #print("theta", theta)
    # Compute scale from mean of two stable mean calculations
    scale = numpy.mean(R[numpy.array([[True, False], [False, True]])]/math.cos(theta));
    #print("scale",scale)
    # % Translation remains the same:
    translation = H[2, 0:2]
    # print("translation", translation)
    HsRt = numpy.zeros((3,3), dtype=numpy.float32)
    HsRt[2,2] = 1

    HsRt[0:2,0:2] = numpy.dot(scale, [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    HsRt[2,0] = translation[0]
    HsRt[2,1] = translation[1]

    return HsRt


if __name__ == "__main__":

    filename = '../input/train.webm'

    last_frame, last_frame_kp, last_frame_descriptors = None, None, None
    corrected_mean, mov_mean = None, None

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(filename)

    count = 0
    Hcumulative = numpy.identity(3)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    while (cap.isOpened()):
        ret, frame = cap.read()

        if frame is not None:

            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # find the keypoints and descriptors with SIFT
            curr_frame_kp, curr_frame_descriptors = sift.detectAndCompute(current_frame, None)

            if count == 0:
                last_frame = current_frame
                last_frame_kp = curr_frame_kp
                last_frame_descriptors = curr_frame_descriptors
                mov_mean = last_frame.astype(numpy.float32)
                corrected_mean = last_frame.astype(numpy.float32)
                count += 1
                continue

            mov_mean = mov_mean + current_frame.astype(numpy.float32)

            # TODO: OPENCV MACHING FUNCTIONS - SHOULD BE REMOVED
            # BFMatcher with default params
            # bf = cv2.BFMatcher()
            # matches = bf.knnMatch(curr_frame_descriptors, last_frame_descriptors, k=2)
            #
            # # store all the good matches as per Lowe's ratio test.
            # good = []
            # for m, n in matches:
            #     if m.distance < 0.7 * n.distance:
            #         good.append(m)
            # src_pts = numpy.float32([curr_frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
            # dst_pts = numpy.float32([last_frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 2)

            matches = match_symmetric(last_frame_descriptors, curr_frame_descriptors)
            # Return the indices of the elements that are non-zero.
            ndx = matches.nonzero()[0]
            # print(ndx)
            src_pts = []
            for id_ in ndx:
                src_pts.append(last_frame_kp[id_].pt)
            src_pts = numpy.asarray(src_pts)

            ndx2 = [int(matches[i]) for i in ndx]

            dst_pts = []
            for id_ in ndx2:
                dst_pts.append(curr_frame_kp[id_].pt)
            dst_pts = numpy.asarray(dst_pts)

            fp = convert_to_homogeneous_coord(numpy.asarray(src_pts))
            tp = convert_to_homogeneous_coord(numpy.asarray(dst_pts))

            H, mask = H_from_ransac(tp, fp, match_theshold=10)
            # TODO: The smoothing part is not working
            #HsRt = convert_affine_to_scale_rotation_trainlation_transform(H)
            #Hcumulative = numpy.dot(HsRt, Hcumulative)
            # Feed those pairs into cvFindHomography to compute the homography between those frames
            # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            corrected_frame = cv2.warpAffine(current_frame, H[0:2, 0:3], (current_frame.T.shape)) # cv2.warpPerspective(current_frame, H, current_frame.T.shape)

            cv2.imshow('frame', last_frame)
            cv2.imshow('corrected', corrected_frame)

            corrected_mean += corrected_frame.astype(numpy.float);

            last_frame = current_frame
            last_frame_kp = curr_frame_kp
            last_frame_descriptors = curr_frame_descriptors
            count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    corrected_mean = corrected_mean / (count);
    mov_mean = mov_mean / (count);
    plot_image = numpy.concatenate((mov_mean, corrected_mean), axis=1)
    plt.imshow(plot_image, cmap="gray")
    plt.show()