import cv2 as cv
import numpy as np
import sys


def help():
    print("\nThis program demonstrated the use of the morphology operations\n"
           "Usage:\n"
           "./morpholgy [4 parameters] \n"
           "where parameters are:\n"
           "1) [image_name -- ]\n"
           "2) [operator -- 0 (Erode) 1 (dilate) 2 (opening) 3 (closing)]\n"
           "3) [element structurant -- 0 (rectangle) 1 (cross) 2 (ellipse)]\n"
           "4) [kernel size (integer value)]")
 

def morphologyOperation(src, morph_operator, morph_element, morph_size) :
    """
    function morphologyOperation
    
    Parameters
    ----------
        src : Mat
            input image
        dst : Mat
            output image
        morph_operator : MorphTypes
            Type of morphological operation
        morph_element : MorphShapes
            Shape of the structuring element
        morph_size : int
            Size of the morphological element
    """

  # Name of the operator
    match int(morph_operator):
        case 0:
            operator_name = "Erode"
            morph_operator = cv.MORPH_ERODE
        case 1:
            operator_name = "Dilate"
            morph_operator = cv.MORPH_DILATE
        case 2:
            operator_name = "Opening"
            morph_operator = cv.MORPH_OPEN
        case 3:
            operator_name = "Closing"
            morph_operator = cv.MORPH_CLOSE
        case _:
            operator_name = "UNKNOWN"
            morph_operator = cv.MORPH_ERODE

     # Name of the structural element
    match int(morph_element):
        case 0:
            element_name = "Rectangle"
            morph_element = cv.MORPH_RECT
        case 1:
            element_name = "Cross"
            morph_element = cv.MORPH_CROSS
        case 2:
            element_name = "Ellipse"
            morph_element = cv.MORPH_ELLIPSE
        case _:
            element_name = "UNKNOWN"
            morph_element = cv.MORPH_RECT

    kernel = cv.getStructuringElement(morph_element, (int(morph_size), int(morph_size)))
    print(kernel)
    
    #Apply the specified morphology operation
    dst = cv.morphologyEx( src, morph_operator, kernel)
    
    window_name = "Morphology-"
    
    kernel_name = str(morph_size)
    
    window_name += operator_name + element_name + kernel_name
    
    cv.imshow(window_name, dst)


def main():

    morph_elem = cv.MORPH_RECT
    morph_size = 3
    morph_operator = cv.MORPH_ERODE

    if len(sys.argv) != 5 :
        print("This program needs 4 parameters!")
        help()
        exit(0)

    filename = sys.argv[1]
    print("File Name: "+filename)
    
    # Read the input image in grayscale
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    
    if src is None :
        help()
        print("Cannot read image file: %s\n", filename)
        exit(1)

    # Convert it to binary
    thresh = cv.threshold(src,127,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    cv.imshow("Original image converted to binary", thresh);
    
    # Getting parameters
    morph_operator = sys.argv[2]
    morph_elem     = sys.argv[3]
    morph_size     = sys.argv[4]
    
    print("Morph operator: "+morph_operator)
    print("Morph element: "+morph_elem)
    print("Kernel Size: "+morph_size)
    
    # Calling the operation
    morphologyOperation(thresh, morph_operator, morph_elem, morph_size)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()