// Giacomello Mattia
// I.D. 1210988

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "filter.h"


// Definition of auxiliary functions
void showHistogram(std::vector<cv::Mat>&);
void equalizeHSVChannels(cv::Mat,std::vector<cv::Mat>& );
void equalizeBGRChannels(cv::Mat,cv::Mat&);
void calculateBGRHistograms(cv::Mat,std::vector<cv::Mat>& );

void sizeGaussian(int, void*);
void sigmaGaussian(int, void*);
void sizeMedian(int, void*);
void rangeBilateral(int, void* );
void spaceBilateral(int, void* );



std::string winNameGaus = "Gaussian Filter";
std::string winNameMed = "Median Filter";
std::string winNameBil = "Bilateral Filter";

cv::String imagePath = "..\\..\\data\\lena.png";


int main()
{    
	/*****************************************************************
	 * First Part: Histogram Equalization on BGR and HSV Color Spaces*
	 *****************************************************************/
	
    std::vector<cv::Mat> histograms;
    // Loading and visualization of input image and RGB histograms associated
    cv::Mat originalImage = cv::imread(imagePath);
    calculateBGRHistograms(originalImage, histograms);

    cv::namedWindow("imageWindow");
    cv::setWindowTitle("imageWindow","Original image");
    cv::imshow("imageWindow",originalImage);
    showHistogram(histograms);
    cv::waitKey();
    
    // RGB histogram equalization and visualization the equalized image and RGB histograms associated
    cv::Mat equalizedBGRImage;
    equalizeBGRChannels(originalImage, equalizedBGRImage);
    calculateBGRHistograms(equalizedBGRImage, histograms);
    cv::setWindowTitle("imageWindow","RGB equalized image");
    cv::imshow("imageWindow",equalizedBGRImage);
    showHistogram(histograms);
    cv::waitKey();
    
    // HSV histogram equalization, one channel at time, and visualization the equalized image and RGB histograms associated
    std::vector<cv::Mat> equalizedHSVImages;
    equalizeHSVChannels(originalImage, equalizedHSVImages);

    std::vector<cv::String> windowTitles = {"H equalized image","S equalized image","V equalized image"};
    for(int i=0 ; i<3; i++){
        calculateBGRHistograms(equalizedHSVImages[i], histograms);
        cv::setWindowTitle("imageWindow",windowTitles[i]);
        cv::imshow("imageWindow",equalizedHSVImages[i]);
        showHistogram(histograms);
        cv::waitKey();
    }

    cv::destroyAllWindows();

    
	/********************************
	 * Second Part: Image Filtering *
	 ********************************/	
    
    // Initialization and maximum value of trackbars
    int initGausSize = 7,
        initGausSigma = 1,
        maxGausSize = 51,
        maxGausSigma = 25;
    
    int initMedSize = 11,
        maxMedSize = 51;
    
    int initBilRange = 5,
        initBilSpace = 1,
        maxBilRange = 500,
        maxBilSpace = 25;

    // Creation of window and trackbar for gaussian filter 
    GaussianFilter Gaussian(equalizedHSVImages[2], initGausSize, initGausSigma);
    sizeGaussian(initGausSize,&Gaussian);
    cv::createTrackbar("Size", winNameGaus, &initGausSize, maxGausSize, sizeGaussian, &Gaussian);
    cv::createTrackbar("Sigma", winNameGaus, &initGausSigma, maxGausSigma, sigmaGaussian, &Gaussian);

    // Creation of window and trackbar for median filter 
    MedianFilter Median(equalizedHSVImages[2], initMedSize);
    sizeMedian(initMedSize, &Median);
    cv::createTrackbar("Size", winNameMed, &initMedSize, maxMedSize, sizeMedian, &Median);

    // Creation of window and trackbar for bilinear filter 
    BilateralFilter Bilateral(equalizedHSVImages[2], initBilSpace * 6, initBilSpace, initBilRange);
    spaceBilateral(initBilSpace, &Bilateral);
    cv::createTrackbar("S. Space", winNameBil, &initBilSpace, maxBilSpace, spaceBilateral, &Bilateral);
    cv::createTrackbar("S. Range", winNameBil, &initBilRange, maxBilRange, rangeBilateral, &Bilateral);

    cv::waitKey();
    return 0;
}



/**
* Method to compute HSV equalized images one channel for image
* @param src Image input   
* @param equalizedBGRImages Output vector of HSV equalized images one channel for image
*/
void equalizeHSVChannels(cv::Mat src, std::vector<cv::Mat>& equalizedBGRImages)
{
    cv::Mat HSVImage;
    std::vector<cv::Mat> HSVChannels(3);
	cv::cvtColor(src, HSVImage, cv::COLOR_BGR2HSV);
    cv::split(HSVImage, HSVChannels);

    equalizedBGRImages = std::vector<cv::Mat>(3);

    for(int i=0;i<3;i++)
    {
        // We make a copy of the vector of the channels and then we substitute the equalized corrisponding channel
        
        std::vector<cv::Mat> equalizedHSVChannels = HSVChannels;
        cv::Mat equalizedHSVChannelTmp;
        cv::equalizeHist(HSVChannels[i], equalizedHSVChannelTmp);
        equalizedHSVChannels[i]=equalizedHSVChannelTmp;

        cv::Mat equalizedHSVImage;
        cv::merge(equalizedHSVChannels, equalizedHSVImage);   
        
        cv::cvtColor(equalizedHSVImage, equalizedBGRImages[i], cv::COLOR_HSV2BGR);
    }
}

/**
* Method to compute the BGR equalize image 
* @param src Image input  
* @param dst Output equalized image
*/
void equalizeBGRChannels(cv::Mat src, cv::Mat& dst)
{
    
    std::vector<cv::Mat> channelsBGRPre(3), channelsBGRPost(3);
    cv::split(src, channelsBGRPre);
    for (int i = 0; i < 3; i++)
    {
        // Histhogram equalization of BGR channels
        cv::equalizeHist(channelsBGRPre[i], channelsBGRPost[i]);
    }
    // Merge of BGR channels equalized
    cv::merge(channelsBGRPost, dst);
}

/**
* Method to compute the BGR histograms 
* @param src Image input  
* @param BGRHistograms Output vector of histograms
*/
void calculateBGRHistograms(cv::Mat src, std::vector<cv::Mat>& BGRHistograms)
{
    // Histogram equalization parameters
    const int size = 256;
    const float range[] = { 0,256 };
    const float* histRange = { range };

    std::vector<cv::Mat> channelsPre(3);
    BGRHistograms = std::vector<cv::Mat>(3);
    cv::split(src, channelsPre);

    for (int i = 0; i < 3; i++)
    {
        // Computation of BGR histograms
        cv::calcHist(&channelsPre[i], 1, 0, cv::Mat(), BGRHistograms[i], 1, &size, &histRange);
    }
}



void sizeGaussian(int newSize, void* GaussianParsed)
{
    GaussianFilter* GaussianFilt = (GaussianFilter*)GaussianParsed;
    GaussianFilt->setSize(newSize);
    GaussianFilt->doFilter();
    cv::imshow(winNameGaus, GaussianFilt->getResult());
}

void sigmaGaussian(int newSigma, void* GaussianParsed)
{
    GaussianFilter* GaussianFilt = (GaussianFilter*)GaussianParsed;
    GaussianFilt->setSigma(newSigma);
    GaussianFilt->doFilter();
    cv::imshow(winNameGaus, GaussianFilt->getResult());
}

void sizeMedian(int newSize, void* MedianParsed)
{
    MedianFilter* MedianFilt = (MedianFilter*)MedianParsed;
    MedianFilt->setSize(newSize);
    MedianFilt->doFilter();
    cv::imshow(winNameMed, MedianFilt->getResult());
}

void spaceBilateral(int newSpace, void* BilateralParsed)
{
    BilateralFilter* BilateralFilt = (BilateralFilter*)BilateralParsed;
    BilateralFilt->setSigmaSpace(newSpace);
    BilateralFilt->setSize(newSpace*6);
    BilateralFilt->doFilter();
    cv::imshow(winNameBil, BilateralFilt->getResult());
}

void rangeBilateral(int newRange, void* BilateralParsed)
{
    BilateralFilter* BilateralFilt = (BilateralFilter*)BilateralParsed;
    BilateralFilt->setSigmaRange(newRange);
    BilateralFilt->doFilter();
    cv::imshow(winNameBil, BilateralFilt->getResult());
}




// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat>& hists)
{
  // Min/Max computation
  double hmax[3] = {0,0,0};
  double min;
  cv::minMaxLoc(hists[0], &min, &hmax[0]);
  cv::minMaxLoc(hists[1], &min, &hmax[1]);
  cv::minMaxLoc(hists[2], &min, &hmax[2]);

  std::string wname[3] = { "blue", "green", "red" };
  cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                           cv::Scalar(0,0,255) };

  std::vector<cv::Mat> canvas(hists.size());

  // Display each histogram in a canvas
  for (int i = 0, end = hists.size(); i < end; i++)
  {
    canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

    for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++)
    {
      cv::line(
            canvas[i],
            cv::Point(j, rows),
            cv::Point(j, rows - (hists[i].at<float>(j) * rows/hmax[i])),
            hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
            1, 8, 0
            );
    }

    cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
  }
}