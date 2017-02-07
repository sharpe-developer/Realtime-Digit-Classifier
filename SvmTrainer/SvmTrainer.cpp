/******************************************************************************

    FILENAME:       SvmTrainer.cpp

    DESCRIPTION:    Routines training an OpenCV SVM model using MNIST images

    AUTHOR:         David Sharpe

    DEPENDENCIES:   OpenCV 2.4.x

******************************************************************************/
#include "opencv2/opencv.hpp"
#include "HogSvm.h"

#include <iostream>
#include <fstream>

using namespace cv;

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Concatenate multiple images into a single image and display
//
// PARAMETERS:
//  images - vector of image matrices to build display image from
//  numImages - number of images to display
//
///////////////////////////////////////////////////////////////////////////////
void DisplayImages(const std::vector<Mat> &images, unsigned int numImages)
{
    //Place images side by side into a single image for display
    Mat sampleImages = images[0];
    for (unsigned int i = 1; i < numImages; i++)
    {
        hconcat(sampleImages, images[i], sampleImages);
    }

    //Show sample images
    const std::string windowName = "Sample Images";
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, sampleImages);
    waitKey(1);
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Read integer stored in big endian from a file
//
// PARAMETERS:
//  file - file to read from
//
// RETURNS:
//  Integer value in machine's native format (big or little endian)
//
///////////////////////////////////////////////////////////////////////////////
unsigned int ReadInt(std::ifstream &file)
{
    char ch = 0;
    unsigned int value = 0;
    for (int i = sizeof(value)-1; i >= 0; i--)
    {
        file.get(ch);
        value |= (ch & 0xFF) << (8 * i);
    }

    return value;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Read the data from the provided MNIST file into a matrix and store each 
//  image matrix in a vector 
//
// PARAMETERS:
//  filename - path to the MNIST file
//  images - reference to return vector of MNIST image matrices
//
// RETURNS:
//  true if MNIST file loaded successfully
///////////////////////////////////////////////////////////////////////////////
bool ReadMnistImageFile(const char *filename, std::vector<Mat> &images)
{      
    //Open the file
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open() == false)
    {
        std::cout << "Failed to open file: " << filename << std::endl;
        return false;
    }

    //Read file header
    unsigned int magicNumber = ReadInt(file);
    unsigned int numItems    = ReadInt(file);

    //Read remaining file header
    unsigned int numRows    = ReadInt(file);
    unsigned int numColumns = ReadInt(file);

    //Compute the total image size
    unsigned int imageSize = numRows * numColumns;    

    //Read each image into a matrix
    for (unsigned int i = 0; i < numItems; i++)
    {
        //Read image data and store in vector 
        Mat image(numRows, numColumns, CV_8UC1);
        file.read(reinterpret_cast<char*>(image.data), imageSize);

        //Convert to binary
        threshold(image, image, 90, 255, THRESH_BINARY);

        images.push_back(image);
    }

    file.close();

#ifdef _DEBUG 
    //Display images for debug    
    DisplayImages(images, 10);    
#endif

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Read the data from the provided MNSIT file into the supplied matrix. 
//  All labels are loaded into the single matrix with one label per 
//  row in the matrix.
//
// PARAMETERS:
//  filename - path to the MNSIT file
//  labels - reference to return the MNSIT labels
//
// RETURNS:
//  true if MNIST file loaded successfully
///////////////////////////////////////////////////////////////////////////////
bool LoadMnistLabelFile(const char *filename, Mat &labels)
{
    //Open the file
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open() == false)
    {
        std::cout << "Failed to open file: " << filename << std::endl;
        return false;
    }

    //Read file header         
    unsigned int magicNumber = ReadInt(file);
    unsigned int numItems    = ReadInt(file);

    //Labels are 1 byte
    //Build matrix for storing image labels
    unsigned int itemSize = 1;
    labels.create(numItems, itemSize, CV_8UC1);

    //Read each label into a Mat row
    for (unsigned int i = 0; i < numItems; i++)
    {
        file.read(reinterpret_cast<char*>(labels.row(i).data), itemSize);
    }

    file.close();

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Load the MNIST training and test images/labels into matrices
//
// PARAMETERS:
//  trainImages - reference to return the MNSIT training images
//  trainLabels - reference to return the MNSIT training labels
//  testImages - reference to return the MNSIT test images
//  testLabels - reference to return the MNSIT test labels
//
// RETURNS:
//  true if all MNIST files loaded successfully
///////////////////////////////////////////////////////////////////////////////
bool LoadMnistData(std::vector<Mat> &trainImages, Mat &trainLabels, 
                   std::vector<Mat> &testImages, Mat &testLabels)
{
    if (ReadMnistImageFile("./data/MNIST/train-images.idx3-ubyte", trainImages) == false)
    {
        std::cout << "Failed to load training images" << std::endl;
        return false;
    }

    if (LoadMnistLabelFile("./data/MNIST/train-labels.idx1-ubyte", trainLabels) == false)
    {
        std::cout << "Failed to load training labels" << std::endl;
        return false;
    }

    if (ReadMnistImageFile("./data/MNIST/t10k-images.idx3-ubyte", testImages) == false)
    {
        std::cout << "Failed to load test images" << std::endl;
        return false;
    }

    if (LoadMnistLabelFile("./data/MNIST/t10k-labels.idx1-ubyte", testLabels) == false)
    {
        std::cout << "Failed to load test labels" << std::endl;
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Convert the MNIST training and test images/labels into a new data 
//  set of digit and non-digit examples
//
// PARAMETERS:
//  trainImages - reference to return the training images
//  trainLabels - reference to return the training labels
//  testImages - reference to return the test images
//  testLabels - reference to return the test labels
//
// RETURNS:
//  true if data set loaded successfully
///////////////////////////////////////////////////////////////////////////////
bool CreateDigitDetectorData(std::vector<Mat> &trainImages, Mat &trainLabels,
    std::vector<Mat> &testImages, Mat &testLabels)
{
    //For all training labels
    for (int i = 0; i < trainLabels.rows; ++i)
    {        
        //Set label to indicate image is a digit
        trainLabels.row(i) = 1;
    }

    //Add non-digit example training images
    for (int i = 0; i < 30000; ++i)
    {
        //Read image file
        const std::string filename = "./data/NotDigits/train/image" + std::to_string(i) + ".bmp";
        trainImages.emplace_back(imread(filename, CV_LOAD_IMAGE_GRAYSCALE));
        if (trainImages.back().data == nullptr)
        {
            std::cout << "Non-digit training image not found: " << filename << std::endl;
            return false;
        }

        //Convert to binary
        threshold(trainImages.back(), trainImages.back(), 90, 255, THRESH_BINARY);

        //Label as non-digit
        trainLabels.push_back(Mat::zeros(1, 1, trainLabels.type()));
    }

    //For all test labels
    for (int i = 0; i < testLabels.rows; ++i)
    {
        //Set label to indicate image is a digit
        testLabels.row(i) = 1;        
    }

    //Add non-digit example test images
    for (int i = 0; i < 10000; ++i)
    {
        //Read image file
        const std::string filename = "./data/NotDigits/test/image" + std::to_string(i) + ".bmp";
        testImages.emplace_back(imread(filename, CV_LOAD_IMAGE_GRAYSCALE));
        if (testImages.back().data == nullptr)
        {
            std::cout << "Non-digit training image not found: " << filename << std::endl;
            return false;
        }

        //Convert to binary
        threshold(testImages.back(), testImages.back(), 90, 255, THRESH_BINARY);

        //Label as non-digit
        testLabels.push_back(Mat::zeros(1, 1, trainLabels.type()));
    }

    return true;
}

int main(int argc, char** argv)
{
    std::vector<Mat> trainImages;
    std::vector<Mat> testImages;
    Mat trainLabels;    
    Mat testLabels;  

    //Load the data from the MNSIT training and test files
    if (LoadMnistData(trainImages, trainLabels, testImages, testLabels) == true)
    {
        try
        {
            ////////////////////////////////////////////////////////////////
            // Train a classifier for handwritten digits
            ////////////////////////////////////////////////////////////////
            HogSvm digitSvm;

            // Set up SVM parameters    
            digitSvm.SetType(ml::SVM::C_SVC);
            digitSvm.SetKernel(ml::SVM::POLY);
            digitSvm.SetGamma(0.1);
            digitSvm.SetDegree(2);
            digitSvm.SetC(0.1);

            //Train the SVM
            std::cout << "Training classification SVM (this will take several minutes)..." << std::endl;
            digitSvm.Train(trainImages, trainLabels);
            
            //Test the SVM
            std::cout << "Classification SVM training complete" << std::endl 
                      << "Testing classification SVM..." << std::endl;            
            float percentError = digitSvm.Test(testImages, testLabels);
            
            //Display results of test
            std::cout << "Classification SVM testing completed. Percent error: " 
                      << percentError << "%" << std::endl;

            //Save the SVM model to a file
            digitSvm.Save("mnistSvm.xml");

            ////////////////////////////////////////////////////////////////
            //Train a detector to determine if an image has a digit or not
            ////////////////////////////////////////////////////////////////
            if (CreateDigitDetectorData(trainImages, trainLabels, testImages, testLabels) == true)
            {
                HogSvm digitDetector;

                // Set up SVM parameters
                digitDetector.SetType(ml::SVM::C_SVC);
                digitDetector.SetKernel(ml::SVM::LINEAR);
                digitDetector.SetC(0.1);

                //Train the SVM
                std::cout << "Training detector SVM (this will take several minutes)..." << std::endl;
                digitDetector.Train(trainImages, trainLabels);

                //Test the SVM
                std::cout << "Detector SVM training complete" << std::endl 
                          << "Testing Detector SVM..." << std::endl;
                float percentError = digitDetector.Test(testImages, testLabels);

                //Display results of test
                std::cout << "Detector SVM testing completed. Percent error: " 
                          << percentError << "%" << std::endl;

                //Save the SVM model to a file
                digitDetector.Save("svmDigitDetector.xml");
            }
        }
        catch (Exception &e)
        {
            std::cout << "OpenCV Exception: " << e.what() << std::endl;
        }
    }

    return 0;
}