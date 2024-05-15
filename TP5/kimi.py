import cv2 as cv
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# ... (other functions from your code)
# Keycode definitions
ESC_KEY = 27
Q_KEY = 113

def parse_command_line_arguments():# Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kp", default="SIFT", help="key point (or corner) detector: GFTT ORB SIFT ")
    parser.add_argument("-n", "--nbKp", default=20, type=int, help="Number of key point required (if configurable) ")
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
    
def feature_matcher(descriptor1, descriptor2, matcher_method='NORM_L2'):
    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptor1, descriptor2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def apply_homography(src_points, dst_points, H, width, height):
    # Convert points to homogeneous coordinates
    src_points = np.array(src_points, dtype='float32')
    src_points = np.hstack([src_points, np.ones((src_points.shape[0], 1))])
    dst_img = np.zeros((height, width, 1), dtype='uint8')
    
    # Apply homography to each source point
    for point in src_points:
        transformed_point = np.dot(H, point)
        transformed_point = transformed_point / transformed_point[2]
        dst_x, dst_y = int(transformed_point[0]), int(transformed_point[1])
        if 0 <= dst_x < width and 0 <= dst_y < height:
            dst_img[dst_y, dst_x] = 255
    
    return dst_img

def apply_homography(img, H):
    # 使用透视变换应用单点变换
    transformed_img = cv.warpPerspective(img, H, (img.shape[1] + img.shape[1], img.shape[0]))
    return transformed_img

def main():
    parser = parse_command_line_arguments()
    args = vars(parser.parse_args())

    # Load the two input images
    img1, gray1 = load_gray_image(args["image1"])
    img2, gray2 = load_gray_image(args["image2"])

    # Display the input images
    display_image(img1, "Image 1")
    display_image(img2, "Image 2")

    # Apply the chosen feature detector
    if args["kp"] == "SIFT":
        sift = cv.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(gray1, None)
        kp2, desc2 = sift.detectAndCompute(gray2, None)
    elif args["kp"] == "ORB":
        orb = cv.ORB_create()
        kp1, desc1 = orb.detectAndCompute(gray1, None)
        kp2, desc2 = orb.detectAndCompute(gray2, None)
    else:
        print("Invalid keypoint detector")
        return

    # Display keypoints
    img_kp1 = cv.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img_kp2 = cv.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
    display_image(img_kp1, "Image 1 Keypoints")
    display_image(img_kp2, "Image 2 Keypoints")

     # 匹配描述符
    matches = feature_matcher(desc1, desc2)

    # 选择匹配点对
    matched_idx = [m.queryIdx for m in matches[:10]]  # 选择前10个匹配点的索引
    matched_kp1 = [kp1[i] for i in matched_idx]
    matched_kp2 = [kp2[m.trainIdx] for m in matches[:10]]

    # 计算单点透视变换矩阵
    pts_src = np.array([kp.pt for kp in matched_kp1], dtype='float32')
    pts_dst = np.array([kp.pt for kp in matched_kp2], dtype='float32')
    H, _ = cv.findHomography(pts_src, pts_dst, cv.RANSAC, 5.0)

    # 对第二张图像应用单点透视变换
    transformed_img2 = apply_homography(img2, H)

    # 确保变换后的图像与第一张图像具有相同的数据类型和通道数
    transformed_img2 = cv.cvtColor(transformed_img2, cv.COLOR_BGR2RGB)

    # 创建全景图
    result = cv.hconcat([img1, transformed_img2])

    # Display the resulting panorama
    display_image(result, "Panorama")

    # Wait for user action
    key = cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()