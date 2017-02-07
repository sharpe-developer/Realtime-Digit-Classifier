/******************************************************************************

    FILENAME:       RealtimeDigitClassifier.cpp

    DESCRIPTION:    Application to classify handwritten digits viewed from a 
                    camera in real time and display the predicted value on the 
                    image. Assumes the digits are written in dark color on a 
                    light (preferably white) background.

    AUTHOR:         David Sharpe

    DEPENDENCIES:   OpenCV 2.4.x

******************************************************************************/
#include "opencv2/opencv.hpp"
#include "HogSvm.h"

#include <iostream>

using namespace cv;

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Process an image frame 
//
// PARAMETERS:
//  classifier - digit classifier object reference
//  classifier - digit detector object reference
//  displayFrame - frame displayed to user
//  frame - frame showing processed image
//
///////////////////////////////////////////////////////////////////////////////
void ProcessFrame(const HogSvm &classifier, const HogSvm &detector, Mat &displayFrame, Mat &frame)
{
    //Convert to grayscale, smooth, and binary threshold the image
    cvtColor(frame, frame, CV_BGR2GRAY);
    blur(frame, frame, Size(5, 5));
    threshold(frame, frame, 110, 255, THRESH_BINARY_INV);       

    //Create a region of interest in the center of the frame
    //Anything outside this area will be ignored
    //Size is percentage of frame
    float size = 0.75f;
    Rect roi = Rect(static_cast<int>(frame.cols * (1 - size) / 2.0f), 
                    static_cast<int>(frame.rows * (1 - size) / 2.0f),
                    static_cast<int>(frame.cols * size),
                    static_cast<int>(frame.rows * size));
    rectangle(displayFrame, roi.tl(), roi.br(), Scalar(0, 0, 255));

    //Floodfill outside of roi to eliminate noise at edges of image
    floodFill(frame, roi.tl(), Scalar(0));
    floodFill(frame, Point(roi.x, roi.y + roi.height), Scalar(0)); //bl
    floodFill(frame, roi.br(), Scalar(0));
    floodFill(frame, Point(roi.x + roi.width, roi.y), Scalar(0)); //tr 

    //Close any holes
    morphologyEx(frame, frame, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

    //Find the contours in the roi
    std::vector<std::vector<Point>> contours;
    Mat contourFrame = frame(roi).clone();
    findContours(contourFrame, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    RNG rng;
    Rect boundRect;

    //Perform classification in the image inside each contour's bounding rectangle
    //and draw bounding rectangles with digit value for classified images    
    for (int i = 0; i < contours.size(); i++)
    {
        //Get bounding rectangle for this contour
        boundRect = boundingRect(contours[i]);
        boundRect += Point(roi.x, roi.y);

        //Get the image contained in the bounding rectangle
        Mat image = frame(Rect(boundRect.x, boundRect.y, boundRect.width, boundRect.height)).clone();
            
        //Extract the area inside the bounding rectangle and add black border padding
        //MNIST digits are padded with 4 pixels on each side of a 20 pixel image (4/20 = 0.2)
        const int hpad = static_cast<int>(boundRect.height * 0.2);             
        const int wpad = static_cast<int>(boundRect.width * 0.2);
        copyMakeBorder(image, image, hpad, hpad, wpad, wpad, BORDER_CONSTANT, 0);

        //Dilate to fatten the digit lines
        //dilate(image, image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
            
        //Does the image contain a digit?
        float detect = detector.Predict(image);            
        if (detect > 0)
        {
            //Draw bounding rectangle with random color
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            rectangle(displayFrame, boundRect.tl(), boundRect.br(), color, 2);

            //Use the SVM to classify the digit in the image
            int prediction = static_cast<int>(classifier.Predict(image));

            //Display prediction on the image at the top left of the bounding rectangle
            putText(displayFrame, std::to_string(prediction), boundRect.tl() - Point(0, 5), FONT_HERSHEY_PLAIN, 1.4, Scalar(0, 0, 0));                
        }        
    }
}


int main(int argc, char** argv)
{   
    const char* classifierFilename = "mnistSvm.xml";
    const char* detectorFilename = "svmDigitDetector.xml";
    
    HogSvm classifier;
    HogSvm detector;
    Mat frame;
    Mat processedFrame;
    char c = 0;

    // Load the classifier model 
    if (classifier.Load(classifierFilename) == false)
    {
        std::cout << "Failed to load classifier model file" << std::endl;
        return 1;
    }

    // Load the detector model 
    if (detector.Load(detectorFilename) == false)
    {
        std::cout << "Failed to load detector model file" << std::endl;
        return 1;
    }

    // Grab the first camera on the system
    VideoCapture vidCapture(0);

    // Verify device opened correctly
    if (!vidCapture.isOpened())
    {
        std::cout << "Could not open video capture device " << std::endl;
        return 1;
    }

    // Get resolution of device
    Size vidSize = Size(static_cast<int>(vidCapture.get(CAP_PROP_FRAME_WIDTH)),
                        static_cast<int>(vidCapture.get(CAP_PROP_FRAME_HEIGHT)));

    std::cout << "Frame resolution: Width = " << vidSize.width
              << " Height = " << vidSize.height << std::endl;

    // Define display window names
    const char* WIN_TEST = "Test";
    const char* WIN_DISPLAY = "Display";

    // Create display windows
    namedWindow(WIN_DISPLAY, WINDOW_AUTOSIZE);  
    moveWindow(WIN_DISPLAY, 0, 0);

    namedWindow(WIN_TEST, WINDOW_AUTOSIZE);
    moveWindow(WIN_TEST, vidSize.width, 0);

    // Main loop
    while (true)
    {
        // Get a frame from the video device
        vidCapture >> frame;
        //frame = imread("numbers.bmp");
        if (frame.empty())
        {
            std::cout << "Failed to capture frame" << std::endl;
            break;
        }

        // Copy original image for processing
        processedFrame = frame;

        // Perform some processing on the frame
        ProcessFrame(classifier, detector, frame, processedFrame);

        // Display results
        imshow(WIN_DISPLAY, frame);
        imshow(WIN_TEST, processedFrame);

        // Wait for key press or timeout
        c = (char)waitKey(50);
        if (c == 'Q' || c == 'q')
        {
            std::cout << "Exiting" << std::endl;
            break;
        }
    }

    return 0;
}


