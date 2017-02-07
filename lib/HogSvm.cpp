/******************************************************************************

    FILENAME:       HogSvm.cpp

    DESCRIPTION:    Implementation of SVM using an images histogram of oriented 
                    gradients (HOG) for SVM features
                    
    AUTHOR:         David Sharpe

******************************************************************************/
#include "HogSvm.h"

using namespace cv;


///////////////////////////////////////////////////////////////////////////////
//  Default constructor
///////////////////////////////////////////////////////////////////////////////
HogSvm::HogSvm() :
    m_hog(Size(28, 28), Size(4, 4), Size(2, 2), Size(4, 4), 9)
{

}

///////////////////////////////////////////////////////////////////////////////
//  Destructor
///////////////////////////////////////////////////////////////////////////////
HogSvm::~HogSvm()
{
    
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Use the SVM to predict the class from the supplied image.
//
// PARAMETERS:
//  image - image matrix
//
// RETURNS:
//  Predicted value for the given image
///////////////////////////////////////////////////////////////////////////////
float HogSvm::Predict(const Mat &image) const
{
    //Extract features from image
    Mat features;
    ExtractFeatures(image, features);

    //Predict the class using the SVM model
    return Svm::Predict(features);
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Train the HogSvm using the supplied features and labels.
//
// PARAMETERS:
//  images - vector of image matricies
//  labels - label matrix (one label per row)
//
// RETURNS:
//  true if HogSvm trained successfully
///////////////////////////////////////////////////////////////////////////////
bool HogSvm::Train(const std::vector<Mat> &images, const Mat &labels) const
{       
    //Extract features from images
    Mat features;
    ExtractFeatures(images, features);

    //Train the SVM using the features
    Svm::Train(features, labels);

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Test the current SVM model using the supplied images and labels.
//
// PARAMETERS:
//  images - vector of image matrixes
//  labels - label matrix (one label per row)
//
// RETURNS:
//  Percent error of classification for supplied images and labels
///////////////////////////////////////////////////////////////////////////////
float HogSvm::Test(const std::vector<Mat> &images, const Mat &labels) const
{
    //Extract features from images
    Mat features;
    ExtractFeatures(images, features);

    //Test the SVM using the features
    return Svm::Test(features, labels);
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Extract features for each image in a vector of image matrixes
//
// PARAMETERS:
//  images - vector of image matrixes
//  features - feature matrix (one row of features per image)
//
// RETURNS:
//  true if features matrix is valid
///////////////////////////////////////////////////////////////////////////////
bool HogSvm::ExtractFeatures(const std::vector<Mat> &images, Mat &features) const
{    
    //Extract features from images
    for (const auto & image : images)
    {
        ExtractFeatures(image, features);
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Extract features from a single image
//
// PARAMETERS:
//  images - image matrix
//  features - feature matrix (one row x number of features)
//
// RETURNS:
//  true if features matrix is valid
///////////////////////////////////////////////////////////////////////////////
bool HogSvm::ExtractFeatures(const cv::Mat & image, cv::Mat & features) const
{
    //Get the number of features for the current HOG parameters
    int numFeatures = static_cast<int>(m_hog.getDescriptorSize());

    //Resize input image to match HOG window size
    Mat hogImage;
    resize(image, hogImage, m_hog.winSize);

//#ifdef _DEBUG
//    //Show sample image
//    const std::string windowName = "Sample HOG Image";
//    namedWindow(windowName, WINDOW_AUTOSIZE);
//    imshow(windowName, hogImage);
//    waitKey(1);
//#endif

    //Compute HOG descriptors
    std::vector<float> descriptors(numFeatures);
    m_hog.compute(hogImage, descriptors);

    //Append row vector of HOG descriptors to end of feature matrix
    features.push_back(Mat(descriptors).t());

    return true;
}
