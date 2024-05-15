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
