import cv2 as cv
import numpy as np
import sys

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113


def main():
    # Data structure to store the image
    im = None
    # default name of the image file
    imName= "imagesDeTest/monarch.png"
   
    # If we give an argument then open it instead of the default image
    if len(sys.argv) == 2 :
      imName = sys.argv[1]
   
	# Reading the image (and forcing it to grayscale)
    print("reading image")
    im = cv.imread(imName,cv.IMREAD_GRAYSCALE)
   
    if im is None or im.size == 0 or (im.shape[0] == 0) or (im.shape[1] == 0):
        print("Could not load image !")
        print("Exiting now...")
        exit(1)

    # Creating a window to display some images
    cv.namedWindow("Original image")
	# Displaying the loaded image in the named window
    cv.imshow("Original image", im)
   
   # Based on the code above (and on the TP's subject)
   # You need to add some code here to create a new image
   # and resize manually the original image (the one called 'im')
   # Then you need to display it in a new OpenCV window (see above)


	# Waiting for the user to press ESCAPE before exiting the application	
    key = 0
   
    while key != ESC_KEY and key!= Q_KEY :
        key = cv.waitKey(1)

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()