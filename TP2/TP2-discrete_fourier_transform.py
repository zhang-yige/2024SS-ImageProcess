from __future__ import print_function
import sys

import cv2 as cv
import numpy as np
import os.path as osp


def print_help():
    print('''
    This program demonstrated the use of the discrete Fourier transform (DFT).
    The dft of an image is taken and it's power spectrum is displayed.
    Usage:
    discrete_fourier_transform.py [image_name -- default ../imagesDeTest/monarch.png]''')

def calculateDFTMag(I):
    """
    function calculateDFTMag
    
    Parameters
    ----------
        I : Mat
            input image

    Result
    ------
        magI : Mat
            DFT magnitude of input image
    """
    ## [expand]
    rows, cols = I.shape
    m = cv.getOptimalDFTSize( rows )
    n = cv.getOptimalDFTSize( cols )
    padded = cv.copyMakeBorder(I, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])
    ## [expand]
    ## [complex_and_real]
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv.merge(planes)         # Add to the expanded another plane with zeros
    ## [complex_and_real]
    ## [dft]
    cv.dft(complexI, complexI)         # this way the result may fit in the source matrix
    ## [dft]
    # compute the magnitude and switch to logarithmic scale
    # = > log(1 + sqrt(Re(DFT(I)) ^ 2 + Im(DFT(I)) ^ 2))
    ## [magnitude]
    cv.split(complexI, planes)                   # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv.magnitude(planes[0], planes[1], planes[0])# planes[0] = magnitude
    magI = planes[0]
    ## [magnitude]
    ## [log]
    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv.add(matOfOnes, magI, magI) #  switch to logarithmic scale
    cv.log(magI, magI)
    ## [log]
    ## [crop_rearrange]
    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows/2)
    cy = int(magI_cols/2)

    q0 = magI[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx+cx, 0:cy]     # Top-Right
    q2 = magI[0:cx, cy:cy+cy]     # Bottom-Left
    q3 = magI[cx:cx+cx, cy:cy+cy] # Bottom-Right

    tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp

    tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp
    ## [crop_rearrange]
    ## [normalize]
    cv.normalize(magI, magI, 0, 1, cv.NORM_MINMAX) # Transform the matrix with float values into a
    ## viewable image form(float between values 0 and 1).
    ## [normalize]
    return magI

def rotate_img(img, angle):
    '''
    img   --image
    angle --rotation angle
    return--rotated img
    '''
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    #获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv.getRotationMatrix2D(rotate_center, angle, 1.0)
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated_img = cv.warpAffine(img, M, (new_w, new_h))
    return rotated_img

def process_img(img, rotate_angle):

    rotated_img = rotate_img(img, rotate_angle)
    mag_img = calculateDFTMag(rotated_img)

    cv.imshow("Input Image"       , rotated_img   )    # Show the result
    cv.imshow("spectrum magnitude", mag_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main(argv):

    print_help()

    filename = sys.argv[1] if len(sys.argv) > 1 else 'imagesDeTest/peppers-512.png'

    I = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    if I is None:
        print('Error opening image')
        return -1
    
    magI = calculateDFTMag(I)

    cv.imshow("Input Image"       , I   )    # Show the result
    cv.imshow("spectrum magnitude", magI)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Code à compléter pour :
    # Appliquer une rotation de 45 degrés à l'image (indice : utiliser getRotationMatrix2D et warpAffine)
    # Appliquer la meme rotation en redimensionnant l'image
    # Montrez les deux images résultat
    # Calcul de la DFT (en amplitude seulement) des images tournées
    # Montrer les 2 DFT résultat
    # Calculer la DFT inverse de l'image originale 
    # Calculer la DFT inverse des images tournées

    # I_90 = cv.rotate(I, cv.ROTATE_90_CLOCKWISE)
    # magI_90 = calculateDFTMag(I_90)
    # cv.imshow("Input Image"       , I_90   )    # Show the result
    # cv.imshow("spectrum magnitude angle90", magI_90)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # I_180 = cv.rotate(I, cv.ROTATE_180)
    # magI_180 = calculateDFTMag(I_180)
    # cv.imshow("Input Image"       , I_180   )    # Show the result
    # cv.imshow("spectrum magnitude", magI_180)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # I_270 = cv.rotate(I, cv.ROTATE_90_COUNTERCLOCKWISE)
    # magI_270 = calculateDFTMag(I_270)
    # cv.imshow("Input Image"       , I_270   )    # Show the result
    # cv.imshow("spectrum magnitude", magI_270)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    process_img(I, 45)
    process_img(I, 90)
    process_img(I, 135)
    process_img(I, 180)
    process_img(I, -45)
    process_img(I, -90)
    process_img(I, -135)
    process_img(I, 60)
    process_img(I, 120)
    process_img(I, -60)
    process_img(I, -120)


    # cv.waitKey()

if __name__ == "__main__":
    main(sys.argv[1:])
