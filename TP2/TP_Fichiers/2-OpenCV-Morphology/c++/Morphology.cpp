#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <string>

using namespace cv;
using namespace std;


/// Original OpenCV example code (refactored)
static void help()
{
    printf("\nThis program demonstrated the use of the morpohlogy operations\n"
           "Usage:\n"
           "./morpholgy [4 parameters] \n"
           "where parameters are:\n"
           "1) [image_name -- default ../imagesDeTest/Baboon.jpg]\n"
           "2) [operator -- 0 (Erode) 1 (dilate) 2 (opening) 3 (closing)]\n"
           "3) [element structurant -- 0 (rectangle) 1 (cross) 2 (ellipse)]\n"
           "4) [kernel size (integer value)]");
}

const char* keys =
{
    "{@image|../imagesDeTest/Baboon.jpg|input image file}"
    "{@operator|MORPH_ERODE| Operator for morphology 0 (Erode) 1 (dilate) 2 (opening) 3 (closing)}"
    "{@element|MORPH_RECT| Structural element for morphology 0 (rectangle) 1 (cross) 2 (ellipse)}"
    "{@kernel|3| Kernel Size: of the form 2n+1}"
};



/**
 * @function morphologyOperation
 */
void morphologyOperation(Mat & src, Mat & dst, MorphTypes & morph_operator, MorphShapes & morph_element, int & morph_size) {

    Mat element = getStructuringElement( morph_element, Size(morph_size, morph_size ));
    
    /// Apply the specified morphology operation
    morphologyEx( src, dst, morph_operator, element );
    
    string window_name = "Morphology-";
    string operator_name="";
    string element_name="";
    string kernel_name="";
    
    // Name of the operator
    switch(morph_operator) {
        case MORPH_ERODE:
            operator_name = "Erode";
            break;
        case MORPH_DILATE:
            operator_name = "Dilate";
            break;
        case MORPH_OPEN:
            operator_name = "Opening";
            break;
        case MORPH_CLOSE:
            operator_name = "Closing";
            break;
        default:
            operator_name = "UNKNOWN";
            break;
    }
    
    // Name of the structural element
    switch(morph_element) {
        case MORPH_RECT:
            element_name = "Rectangle";
            break;
        case MORPH_CROSS:
            element_name = "Cross";
            break;
        case MORPH_ELLIPSE:
            element_name = "Ellipse";
            break;
        default:
            element_name = "UNKNOWN";
            break;
    }
    
    
    kernel_name = to_string(morph_size);
    
    window_name += operator_name + element_name + kernel_name;
    
    imshow( window_name, dst );
    
}


/// Main function
int main( int argc, char** argv )
{
    /// Variables
    Mat src, dst;
    
    MorphShapes morph_elem = MORPH_RECT;
    int morph_size = 3;
    MorphTypes morph_operator = MORPH_ERODE;

    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);
    cout <<"File Name: "<<filename<<endl;
    
    /// Read the inpur image in grayscale
    src = imread(filename.c_str(), IMREAD_GRAYSCALE);
    
    /// Convert it to binary
    double thresh = cv::threshold(src, src, 128, 255, THRESH_BINARY | THRESH_OTSU);
    
    if( src.empty() )
    {
        help();
        printf("Cannot read image file: %s\n", filename.c_str());
        return -1;
    }
    
    imshow("Original image converted to binary", src);
    
    /// Getting parameters
    morph_operator = parser.get<MorphTypes>(1);
    morph_elem     = parser.get<MorphShapes>(2);
    morph_size     = parser.get<int>(3);
    
    cout << "Morph operator: "<<morph_operator<<endl;
    cout << "Morph element: "<<morph_elem<<endl;
    cout << "Kernel Size: "<< morph_size<<endl;
    
    // Calling the operation
    morphologyOperation(src, dst, morph_operator, morph_elem, morph_size);
    
    waitKey(0);
    
    destroyAllWindows();
    
    return 0;
}
