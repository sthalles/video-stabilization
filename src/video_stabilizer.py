import numpy as np
import cv2

cap = cv2.VideoCapture('../input/train.webm')
count = 0

MIN_MATCH_COUNT = 50

while (cap.isOpened()):
    ret, frame = cap.read()

    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rows, cols = current_frame.shape

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT for the current frame
    kp1, des1 = sift.detectAndCompute(current_frame, None)

    if count == 0:
        previous_frame, kp2, des2 = current_frame, kp1, des1

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # test if there are enough good feature points
    if len(good) > MIN_MATCH_COUNT:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Feed those pairs into cvFindHomography to compute the homography between those frames
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 30.0)

        dst = cv2.warpPerspective(current_frame, M, (cols, rows))

        # dst = cv2.perspectiveTransform(img1,M[0:2,:],(cols,rows))
        cv2.imshow('frame', dst)
        cv2.imshow('original', current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        previous_frame, kp2, des2 = current_frame, kp1, des1

    else:
        print("Not enough matches are found - %d/%d") % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    count += 1

cap.release()
cv2.destroyAllWindows()