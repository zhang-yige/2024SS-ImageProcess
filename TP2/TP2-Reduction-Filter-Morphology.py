import cv2, os
import numpy as np
import os.path as osp

# 1 Présentation de la structure de données cv : :Mat
def open_grey_image(image_path):
    img_grey = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Grey image', img_grey) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    folder, name = osp.split(image_path)
    save_path = osp.join(folder, osp.splitext(name)[0] + '_grey.png')
    cv2.imwrite(save_path, img_grey)

# 2 Redimensionnement d’image
# resized_img = cv2.resize(img_color, (64, 64))
def resize_image_manually(image, target_size):
    height, width = image.shape[:2]
    tgt_h, tgt_w = target_size
    h_scale = height // tgt_h
    w_scale = width // tgt_w

    resized_image = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)

    for y in range(tgt_h):
        for x in range(tgt_w):
            pixel = image[y*h_scale, x*w_scale]
            resized_image[y, x] = pixel

    return resized_image

# 3 Application de Filtres de Convolution
def filters(image, kernel):
    pass

# 4
# Transformez l’image en une image binaire
def binarize_image(image, threshold):
    gray_image = np.mean(image, axis=2)
    binary_image = np.where(gray_image > threshold, 255, 0).astype(np.uint8)
    cv2.imshow('Binary image', binary_image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return binary_image

def erosion(bin_im, kernel):
    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]

    center_coo = (kernel_w // 2, kernel_h // 2)
    assert kernel[center_coo[0], center_coo[1]] != 0

    erode_img = np.zeros(shape=bin_im.shape)
    for i in range(center_coo[0], bin_im.shape[0]-kernel_w+center_coo[0]+1):
        for j in range(center_coo[1], bin_im.shape[1]-kernel_h+center_coo[1]+1):
            a = bin_im[i-center_coo[0]:i-center_coo[0]+kernel_w,
                j-center_coo[1]:j-center_coo[1]+kernel_h]  # 找到每次迭代中对应的目标图像小矩阵
            if np.sum(a * kernel) > 0:  
                erode_img[i, j] = 255

    cv2.imshow('Erosion image', erode_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return erode_img


def dilation(bin_im, kernel):
    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]
    center_coo = (kernel_w // 2, kernel_h // 2)

    assert kernel[center_coo[0], center_coo[1]] != 0
    
    dilate_img = np.zeros_like(bin_im)
    
    for i in range(center_coo[0], bin_im.shape[0] - kernel_w + center_coo[0] + 1):
        for j in range(center_coo[1], bin_im.shape[1] - kernel_h + center_coo[1] + 1):

            overlap = bin_im[i - center_coo[0]:i - center_coo[0] + kernel_w,
                             j - center_coo[1]:j - center_coo[1] + kernel_h]

            dilate_img[i, j] = np.max(overlap * kernel)

    cv2.imshow('Dilation image', dilate_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return dilate_img


def img_open(image, structuring_element1, structuring_element2):
    open_img = erosion(image, structuring_element1)
    open_img = dilation(open_img, structuring_element2)

    cv2.imshow('Open image', open_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return open_img

def img_close(image, structuring_element1, structuring_element2):
    close_img = dilation(image, structuring_element1)
    close_img = erosion(close_img, structuring_element2)

    cv2.imshow('Close image', close_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return close_img

def main():
    # 1
    image_path = "imagesDeTest/monarch.png"
    open_grey_image(image_path)

    # 2 Redimensionnement d’image
    image256_path = 'imagesDeTest/peppers-256-RGB.png'
    img256 = cv2.imread(image256_path)
    assert img256.shape[:2] == (256, 256)
    target_size = (64, 64)
    resized_img = resize_image_manually(img256, target_size)
    folder, name = osp.split(image256_path)
    save_path = osp.join(folder, osp.splitext(name)[0] + '_resized_to_64.png')
    cv2.imwrite(save_path, resized_img)

    # 3 Application de Filtres de Convolution
    img = cv2.imread("imagesDeTest/peppers-512.png")

    save_dir = 'imagesDeTest/filters'
    os.makedirs(save_dir, exist_ok=True)
    kernel = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]]) /16
    dst = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(osp.join(save_dir, '1.png'), dst)

    kernel = np.ones([3,3]) /9
    dst = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(osp.join(save_dir, '2.png'), dst)

    kernel = np.array([[1, -3, 1],
                      [-3, 9, -3],
                      [1, -3, 1]])
    dst = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(osp.join(save_dir, '3.png'), dst)

    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    dst = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(osp.join(save_dir, '4.png'), dst)

    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    dst = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(osp.join(save_dir, '5.png'), dst)

    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    dst = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(osp.join(save_dir, '6.png'), dst)

    # 4 Morphologie Mathématique
    # Transformez l’image en une image binaire
    save_dir = 'imagesDeTest/morphology'
    os.makedirs(save_dir, exist_ok=True)
    threshold_value = 127
    binary_image = binarize_image(img, threshold_value)
    cv2.imwrite(osp.join(save_dir, 'pepper-512-binary_image.png'), binary_image)

    structuring_element = np.ones((5, 5), dtype=np.uint8)
    eroded_image = erosion(binary_image, structuring_element)
    cv2.imwrite(osp.join(save_dir, 'pepper-512-eroded_image-5x5.png'), eroded_image)
    dilated_image = dilation(binary_image, structuring_element)
    cv2.imwrite(osp.join(save_dir, 'pepper-512-dilated_image-5x5.png'), dilated_image)

    open_image = img_open(binary_image, structuring_element, structuring_element)
    cv2.imwrite(osp.join(save_dir, 'pepper-512-open_image-5x5.png'), open_image)
    close_image = img_close(binary_image, structuring_element, structuring_element)
    cv2.imwrite(osp.join(save_dir, 'pepper-512-close_image-5x5.png'), close_image)
    

if __name__ == "__main__":
    main()