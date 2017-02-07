/******************************************************************************

    FILENAME:       Svm.cpp

    DESCRIPTION:    Methods for training, testing, and prediction using an 
                    OpenCV SVM model

    AUTHOR:         David Sharpe

******************************************************************************/
#include "Svm.h"

#include <filesystem>

using namespace cv;
using namespace ml;


///////////////////////////////////////////////////////////////////////////////
//  Default constructor
///////////////////////////////////////////////////////////////////////////////
Svm::Svm()
{
    //Create an SVM model
    m_svm = SVM::create();
}

///////////////////////////////////////////////////////////////////////////////
//  Destructor
///////////////////////////////////////////////////////////////////////////////
Svm::~Svm()
{
    
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Use the SVM to predict the class from the supplied features.
//
// PARAMETERS:
//  features - feature vector (row vector)
//
// RETURNS:
//  Predicted value for given features
///////////////////////////////////////////////////////////////////////////////
float Svm::Predict(const cv::Mat &features) const
{
    //Convert data to format required by SVM
    Mat input;
    features.convertTo(input, CV_32FC1);

    //Predict the class using the SVM model
    return m_svm->predict(input);
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Train the SVM using the supplied features and labels.
//
// PARAMETERS:
//  features - feature matrix (one feature set per row)
//  labels - label matrix (one label per row)
//
// RETURNS:
//  true if SVM trained successfully
///////////////////////////////////////////////////////////////////////////////
bool Svm::Train(const cv::Mat &features, const cv::Mat &labels) const
{
    //Convert data to format required by SVM
    Mat svmFeatures, svmLabels;
    features.convertTo(svmFeatures, CV_32FC1);
    labels.convertTo(svmLabels, CV_32SC1);    

    // Train the SVM model
    m_svm->train(svmFeatures, ROW_SAMPLE, svmLabels);    

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Train the SVM using the supplied features and labels.
//
// PARAMETERS:
//  features - feature matrix (one feature set per row)
//  labels - label matrix (one label per row)
//
// RETURNS:
//  true if SVM trained successfully
///////////////////////////////////////////////////////////////////////////////
bool Svm::TrainAuto(const cv::Mat &features, const cv::Mat &labels) const
{
    //Convert data to format required by SVM
    Mat svmFeatures, svmLabels;
    features.convertTo(svmFeatures, CV_32FC1);
    labels.convertTo(svmLabels, CV_32SC1);

    Ptr<TrainData> td = TrainData::create(svmFeatures, ROW_SAMPLE, svmLabels);

    //TODO add member functions to set the parameters below
    ParamGrid Cgrid(10, 20, 1.1);
    ParamGrid gammaGrid(0.5, 2, 1.1);
    ParamGrid pGrid(0, 0, 0);
    ParamGrid nuGrid(0, 0, 0);
    ParamGrid coeffGrid(0, 0, 0);
    ParamGrid degreeGrid(0, 0, 0);
    bool balanced = false;
    int k = 10;

    // Train the SVM model
    m_svm->trainAuto(td, k, Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid, balanced);  

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Test the current SVM model using the supplied features and labels.
//
// PARAMETERS:
//  features - feature matrix (one item per row)
//  labels - label matrix (one label per row)
//
// RETURNS:
//  Percent error of classification for supplied features and labels
///////////////////////////////////////////////////////////////////////////////
float Svm::Test(const cv::Mat &features, const cv::Mat &labels) const
{
    //Convert data to format required by SVM
    Mat svmFeatures, svmLabels;
    features.convertTo(svmFeatures, CV_32FC1);
    labels.convertTo(svmLabels, CV_32SC1);    

    //Predict each example using the model and compare prediction to actual label
    int result = 0;
    int errors = 0;
    for (int i = 0; i < svmFeatures.rows; i++)
    {
        result = static_cast<int>(m_svm->predict(svmFeatures.row(i)));
        if (result != svmLabels.at<int>(i, 0))
        {
            errors++;
        }
    }

    float percentError = (100.0f * errors) / svmLabels.rows;

    //Display results of test
    //std::cout << "SVM testing completed" << std::endl
    //          << "Number of inputs: " << svmLabels.rows << std::endl
    //          << "Number of errors: " << errors << std::endl
    //          << "Percent error:    " << percentError << "%"
    //          << std::endl;

    return percentError;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Load an SVM model file
//
// PARAMETERS:
//  filename - path to SVM model file
//
// RETURNS:
//  true if model loaded successfully
///////////////////////////////////////////////////////////////////////////////
bool Svm::Load(const std::string &filename)
{
    //Check if the SVM model file exists 
    if (std::experimental::filesystem::exists(filename) == false)
    {
        return false;
    }

    //Load the SVM model
    m_svm = StatModel::load<SVM>(filename);
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Save the current SVM model to a file
//
// PARAMETERS:
//  filename - path for SVM model file
//
// RETURNS:
//  true if model saved successfully
///////////////////////////////////////////////////////////////////////////////
bool Svm::Save(const std::string &filename) const
{
    //Validate filename
    if (filename.empty())
    {
        return false;
    }

    //Save the model
    m_svm->save(filename);
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Set the SVM type
//
// PARAMETERS:
//  type - SVM type
///////////////////////////////////////////////////////////////////////////////
void Svm::SetType(cv::ml::SVM::Types type) const
{
    m_svm->setType(type);
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Set the SVM kernel type
//
// PARAMETERS:
//  kernel - SVM kernel type
///////////////////////////////////////////////////////////////////////////////
void Svm::SetKernel(cv::ml::SVM::KernelTypes kernel) const
{
    m_svm->setKernel(kernel);
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Set the SVM termination criteria
//
// PARAMETERS:
//  termCriteria - SVM termination criteria
///////////////////////////////////////////////////////////////////////////////
void Svm::SetTermCriteria(cv::TermCriteria termCriteria) const
{
    m_svm->setTermCriteria(termCriteria);    
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Set the SVM gamma parameter
//
// PARAMETERS:
//  gamma - SVM gamma parameter
///////////////////////////////////////////////////////////////////////////////
void Svm::SetGamma(double gamma) const
{
    m_svm->setGamma(gamma);    
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Set the SVM C parameter
//
// PARAMETERS:
//  c - SVM c parameter
///////////////////////////////////////////////////////////////////////////////
void Svm::SetC(double c) const
{    
    m_svm->setC(c);
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Set the SVM degree parameter
//
// PARAMETERS:
//  degree - SVM degree parameter
///////////////////////////////////////////////////////////////////////////////
void Svm::SetDegree(double degree) const
{
    m_svm->setDegree(degree);
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Set the SVM nu parameter
//
// PARAMETERS:
//  nu - SVM nu parameter
///////////////////////////////////////////////////////////////////////////////
void Svm::SetNu(double nu) const
{
    m_svm->setNu(nu);
}

///////////////////////////////////////////////////////////////////////////////
// DESCRIPTION:
//  Set the SVM P parameter
//
// PARAMETERS:
//  p - SVM P parameter
///////////////////////////////////////////////////////////////////////////////
void Svm::SetP(double p) const
{
    m_svm->setP(p);
}
