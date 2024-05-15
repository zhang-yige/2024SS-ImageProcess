import cv2 as cv
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113


def parse_command_line_arguments():# Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kp", default="SIFT", help="key point (or corner) detector: GFTT ORB SIFT ")
    parser.add_argument("-n", "--nbKp", default=500, type=int, help="Number of key point required (if configurable) ")
    parser.add_argument("-d", "--descriptor", default=True, type=bool, help="compute descriptor associated with detector (if available)")
    parser.add_argument("-m", "--matching", default="NORM_L1", help="Brute Force norm: NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2")
    parser.add_argument("-i1", "--image1", default="TP5/IMG_1_reduced.jpg", help="path to image1")
    parser.add_argument("-i2", "--image2", default="TP5/IMG_2_reduced.jpg", help="path to image2")
    # other argument may need to be added
    return parser

def test_load_image(img):
    if img is None or img.size == 0 or (img.shape[0] == 0) or (img.shape[1] == 0):
        print("Could not load image !")
        print("Exiting now...")
        exit(1)

def load_gray_image(path):
    if(path != None):
        img = cv.imread(path)
        test_load_image(img)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    else:
        img = None
        gray = None
    return img, gray

def display_image(img, image_window_name):
    cv.namedWindow(image_window_name)
    cv.imshow(image_window_name, img)

def feature_detector(type, gray, nb):
    if gray is not None :
        match type :
            case "GFTT":
                 # TODO
                print("not implemented yet")
                sys.exit(1)    
            case "ORB":
                orb = cv.ORB_create(nb)
                kp = orb.detect(gray, None)
            case _:
                sift = cv.SIFT_create(nb)
                kp=sift.detect(gray, None)
    else:
        kp =  None
    return kp

def feature_extractor(type, img, kp):
    desc = None
    if type == "SIFT":
        sift = cv.SIFT_create()
        kp, desc = sift.compute(img, kp)
    elif type == "ORB":
        orb = cv.ORB_create()
        kp, desc = orb.compute(img, kp)
    elif type == "GFTT":
        # GFTT does not have a built-in descriptor, you might want to use another method or create a custom one
        # Here, we just return the keypoints without descriptors
        pass
    return desc

def calculate_homography(kp1, kp2, matches):
    dst_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
    return H

def warp_and_stitch(img1, img2, H):
    result = cv.warpPerspective(img2, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))
    result[0:img1.shape[0], 0:img1.shape[1]] = img1

    return result

def main():

    parser = parse_command_line_arguments()
    args = vars(parser.parse_args())

    # Load, transform to gray the 2 input images
    print("load image 1")
    img1, gray1 = load_gray_image(args["image1"])
    print("load image 2")
    img2, gray2 = load_gray_image(args["image2"])

    # displays the 2 input images
    if img1 is not None : display_image(img1, "Image 1")
    if img2 is not None : display_image(img2, "Image 2")

    # Apply the choosen feature detector
    print(args["kp"]+" detector")
    
    kp1 = feature_detector(args["kp"], gray1, args["nbKp"])
    desc1 = feature_extractor(args["kp"], gray1, kp1)
    if img2 is not None: 
        kp2 = feature_detector(args["kp"], gray2, args["nbKp"])
        desc2 = feature_extractor(args["kp"], gray2, kp2)

    # Display the keyPoint on the input images
    img_kp1=cv.drawKeypoints(gray1,kp1,img1)
    if img2 is not None: img_kp2=cv.drawKeypoints(gray2,kp2,img2)
    
    display_image(img_kp1, "Image 1 "+args["kp"])
    if img2 is not None : display_image(img_kp2, "Image 2 "+args["kp"])

    # code to complete (using functions):
    # - to extract feature and compute descriptor with ORB and SIFT 
    # - to calculate brute force matching between descriptor using different norms
    # - to calculate and apply homography 
    # - to stich and display resulting image

    # match
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(desc1,desc2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    res = cv.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    display_image(res, "matches 1")

    # KNN match
    bf = cv.BFMatcher()
    knn_matches = bf.knnMatch(desc1,desc2, k=2)
    matches = []
    for m,n in knn_matches:
        if m.distance < 0.75*n.distance:
            matches.append([m])
    res = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
    display_image(res, "matches 2 knn")

    H = calculate_homography(kp1, kp2, matches)
    print(H)
    img1, gray1 = load_gray_image(args["image1"])
    img2, gray2 = load_gray_image(args["image2"])
    panorama = warp_and_stitch(img1, img2, H)
    display_image(panorama, "Panorama")

    # waiting for user action
    key = 0
    while key != ESC_KEY and key!= Q_KEY :
        key = cv.waitKey(1)

    # Destroying all OpenCV windows
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()