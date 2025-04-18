#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help(char ** argv)
{
    cout << endl
        <<  "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        <<  "The dft of an image is taken and it's power spectrum is displayed."  << endl << endl
        <<  "Usage:"                                                                      << endl
        << argv[0] << " [image_name -- default ../imagesDeTest/monarch.png]" << endl << endl;
}

Mat calculateDFTMag(Mat I){
//! [expand]
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
//! [expand]

//! [complex_and_real]
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
//! [complex_and_real]

//! [dft]
    dft(complexI, complexI);            // this way the result may fit in the source matrix
//! [dft]

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
//! [magnitude]
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
//! [magnitude]

//! [log]
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);
//! [log]

//! [crop_rearrange]
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
//! [crop_rearrange]

//! [normalize]
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
//! [normalize]
    return magI;
}

int main(int argc, char ** argv)
{
    help(argv);

    const char* filename = argc >=2 ? argv[1] : "../imagesDeTest/monarch.png";

    Mat I = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);
    if( I.empty()){
        cout << "Error opening image" << endl;
        return EXIT_FAILURE;
    }

    Mat magI = calculateDFTMag(I);

    imshow("Input Image"       , I   );    // Show the result
    imshow("spectrum magnitude", magI);


    // Code à compléter pour
    // 1) appliquer une rotation de 45 degrées à l'image (indice : utiliser getRotationMatrix2D et warpAffine)
    // 2) appliquer la meme rotation en redimensionnant l'image
    // Montrez les deux images résultat
    // Calcul de la DFT (en amplitude seulement) des images tournées
    // Montrer les 2 DFT résultat
    // Calculer la DFT inverse de l'image originale
    // Calculer la DFT inverse des images tournées
    
    waitKey();
    
    return EXIT_SUCCESS;
}
