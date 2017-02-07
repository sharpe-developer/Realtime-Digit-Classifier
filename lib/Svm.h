/******************************************************************************

    FILENAME:       Svm.h

    DESCRIPTION:    Methods for training, testing, and prediction using an 
                    OpenCV SVM model

    AUTHOR:         David Sharpe
    
    DEPENDENCIES:   OpenCV 2.4.x

******************************************************************************/
#pragma once

///////////////////////////////////////////////////////////////////////////////
// Include Files
///////////////////////////////////////////////////////////////////////////////
#include "opencv2/opencv.hpp"


///////////////////////////////////////////////////////////////////////////////
// Class Definition
///////////////////////////////////////////////////////////////////////////////
class Svm
{
    ///////////////////////////////////////////////////////////////////////////
    // Construction/Destruction
    ///////////////////////////////////////////////////////////////////////////
public:
    Svm();
    virtual ~Svm();

    ///////////////////////////////////////////////////////////////////////////
    // Public Functions
    ///////////////////////////////////////////////////////////////////////////
public:
    float Predict(const cv::Mat &features) const;
    bool  Train(const cv::Mat &features, const cv::Mat &labels) const;
    bool  TrainAuto(const cv::Mat &features, const cv::Mat &labels) const;
    float Test(const cv::Mat &features, const cv::Mat &labels) const;
    bool  Load(const std::string &filename);
    bool  Save(const std::string &filename) const;
    
    void  SetType(cv::ml::SVM::Types type) const;
    void  SetKernel(cv::ml::SVM::KernelTypes kernel) const;
    void  SetTermCriteria(cv::TermCriteria termCriteria) const;
    void  SetGamma(double gamma) const;
    void  SetC(double c) const;
    void  SetDegree(double degree) const;
    void  SetNu(double nu) const;
    void  SetP(double p) const;


    ///////////////////////////////////////////////////////////////////////////
    // Protected Functions
    ///////////////////////////////////////////////////////////////////////////
protected:

    ///////////////////////////////////////////////////////////////////////////
    // Protected Variables
    ///////////////////////////////////////////////////////////////////////////
protected:
    cv::Ptr<cv::ml::SVM> m_svm;

};