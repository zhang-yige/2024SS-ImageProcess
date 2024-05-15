//
//  main.cpp
//  VSION-TP1
//
//  Created by Jean-Marie Normand on 04/02/2016.
//  Copyright © 2016 Jean-Marie Normand. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// no idea why on the virtual box the keycode is completely wrong
#define ESC_KEY      27
#define ESC_KEY_VM   1048603 // should be 27

#define Q_KEY        113
#define Q_KEY_VM     1048689 // should be 113

using namespace std;
using namespace cv;




int main(int argc, const char * argv[]) {
   // the OpenCV data structure storing the image
	Mat im;
   
   // the (default) name of the image file
   String imName= "../imagesDeTest/monarch.png";
   
   // If we give an argument then open it instead of the default image
   if(argc == 2) {
      imName = argv[1];
   }
   
	// Reading the image (and forcing it to grayscale)
    cout<<"reading image"<<endl;
	im = imread(imName,IMREAD_GRAYSCALE);
   
   if(!im.data || im.empty() || (im.rows == 0)|| (im.cols == 0)) {
      cout << "Could not load image !" <<endl;
      cout << "Exiting now..." << endl;
      return 1;
   }
    
	// Creating a window to display some images
	namedWindow("Original image");
	// Displaying the loaded image in the named window
	imshow("Original image", im);
   

   // Based on the code above (and on the TP's subject)
   // You need to add some code here to create a new image
   // and resize manually the original image (the one called 'im')
   // Then you need to display it in a new OpenCV window (see above)
    
    
    //Resize only if entry image is 256x256
    if(im.cols == 256 && im.rows == 256){
        Mat imrs(64, 64, im.type());
        for (int i = 0; i < imrs.rows; i++) {
            for (int j = 0; j < imrs.cols; j++) {
                imrs.at<uchar>(i, j) = im.at<uchar>(4 * i, 4 * j);
            }
        }
        namedWindow("Resized - 64x64");
        imshow("Resized - 64x64", imrs);
    }
    
    //Filtre Binomial
    Mat kernel = 0.0625*(Mat_<double>(3,3) << 1,2,1,2,4,2,1,2,1);
    filter2D(im, im, -1, kernel);
    namedWindow("Filtre Binomial");
    imshow("Filtre Binomial", im);

    //Filtre Moyenneur
    kernel = Mat::ones(3,3,CV_64F)/9.0;
    filter2D(im, im, -1, kernel);
    namedWindow("Filtre Moyenneur");
    imshow("Filtre Moyenneur", im);

    //Filtre Rehausseur de contraste
    kernel = (Mat_<int>(3, 3) << 1, -3, 1, -3, 9, -3, 1, -3, 1);
    filter2D(im, im, -1, kernel);
    namedWindow("Filtre Rehausseur de contraste");
    imshow("Filtre Rehausseur de contraste", im);

    //Filtre Sobel Horizontal
    kernel = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    filter2D(im, im, -1, kernel);
    namedWindow("Filtre Sobel Horizontal");
    imshow("Filtre Sobel Horizontal", im);

    //Filtre Gradient Oblique 45∞
    kernel = (Mat_<int>(3, 3) << 0,-1,-1,1,0,-1,1,1,0);
    filter2D(im, im, -1, kernel);
    namedWindow("Filtre Gradient Oblique 45°");
    imshow("Filtre Gradient Oblique 45°", im);

    //Filtre Laplacien CarrÈ
    kernel = (Mat_<int>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    filter2D(im, im, -1, kernel);
    namedWindow("Filtre Laplacien Carré");
    imshow("Filtre Laplacien Carré", im);

	// Waiting for the user to press ESCAPE before exiting the application	
   int key = 0;
   
   // you may need to change ESC_KEY to ESC_KEY_VM
   // and Q_KEY to Q_KEY_VM
   while ( (key != ESC_KEY) && (key!= Q_KEY) ) {
		key = waitKey(1);
       
	}

	destroyAllWindows();
   
	return 0;
}
