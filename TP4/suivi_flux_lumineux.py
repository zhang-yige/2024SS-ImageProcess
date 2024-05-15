# SY2324120 Gabriel
# ZY2324111 Hamza

import numpy as np
import cv2 as cv
import argparse
import sys

ESC_KEY = 27
Q_KEY = 113

def main():
    # parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
    #  The example file can be downloaded from: \
    #  https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
    # parser.add_argument('image', type=str, help='path to image file')
    # args = parser.parse_args()

    filename = sys.argv[1] if len(sys.argv) > 1 else 'TP4/video.avi'

    # cap = cv.VideoCapture(args.image)
    cap = cv.VideoCapture(filename)

    run = cap.isOpened()
    # Making sure the capture has opened successfully
    if not run:
        # capture opening has failed we cannot do anything :'(
        print("capture opening has failed we cannot do anything :'(")
        sys.exit()

    # Creating a window to display some images
    cv.namedWindow("Original video")
    cv.namedWindow("Gray video")

    # A key that we use to store the user keyboard input
    key = None

    # fps and total image N

    g_video_vecteur = []
    c_video_vecteur = []

    # Waiting for the user to press ESCAPE before exiting the application
    while key != ESC_KEY and key != Q_KEY:
        ret, im = cap.read()

        # Turning im into grayscale and storing it in imGray

        if ret:
            imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            cv.imshow("Original video", im)
            cv.imshow("Gray video", imGray)
            c_video_vecteur.append(im)
            g_video_vecteur.append(imGray)

        # Look for pollKey documentation
        key = cv.pollKey()
    cv.destroyAllWindows()
    N = cap.get(cv.CAP_PROP_FRAME_COUNT)
    M = 0
    M_input = int(input("please input the number of pictures you need to get the background:"))
    if N:
        if N < M_input:
            M = N
        else:
            M = M_input

    if M > 0:
        average_BGR = np.sum(c_video_vecteur[:M], axis=0) / M
        average_gray = np.sum(g_video_vecteur[:M], axis=0) / M
        background_BGR = average_BGR.astype(np.uint8)
        background_gray = average_gray.astype(np.uint8)
        cv.imshow("Background_BGR", background_BGR)
        cv.imshow("Background_gray", background_gray)
    cv.waitKey(0)

    # Destroying all OpenCV windows
    cv.destroyAllWindows()

    # turn the sky black
    for i in range(average_gray.shape[0]):
        for j in range(average_gray.shape[1]):
            if average_gray[i][j] > 245:
                average_gray[i][j] = 0
    background_sky = average_gray.astype(np.uint8)

    # make the binary image
    _, mask0 = cv.threshold(background_sky, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Execute image opening operation to remove unnecessary areas
    kernel = np.ones((80, 10), np.uint8)
    mask0 = cv.morphologyEx(mask0, cv.MORPH_OPEN, kernel)

    # Perform image closing operations, connect adjacent images, and smooth edges
    kernel_size = (200, 250)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
    mask0 = cv.morphologyEx(mask0, cv.MORPH_CLOSE, kernel)

    kernel = np.ones((50, 50), np.uint8)
    mask0 = cv.morphologyEx(mask0, cv.MORPH_OPEN, kernel)

    kernel = np.ones((60, 50), np.uint8)
    mask0 = cv.morphologyEx(mask0, cv.MORPH_OPEN, kernel)

    cv.imshow("mask", mask0)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cap = cv.VideoCapture(filename)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=mask0, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while (1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.bitwise_and(frame_gray, mask0)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv.add(frame, mask)

            cv.imshow('frame', img)
            k = cv.waitKey(10) & 0xff
            if k == ESC_KEY:
                return
            if k == Q_KEY:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    print("Finished, press any key to close the window...")
    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
