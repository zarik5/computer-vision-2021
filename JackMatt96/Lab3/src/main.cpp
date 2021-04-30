// Giacomello Mattia
// I.D. 1210988


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "filter.h"


// Definition of auxiliary functions
void showHistogram(std::vector<cv::Mat>&);
cv::Mat equalizeSingleChannel(cv::Mat,int);
cv::Mat equalizeBGRChannels(cv::Mat);
std::vector<cv::Mat> calculateHistograms(cv::Mat, int, float* );
void showImagesAndHistogram(std::vector<cv::Mat> ,std::vector<cv::String> , std::vector<std::vector<cv::Mat>> , std::string ,int );

void sizeGaussian(int, void*);
void sigmaGaussian(int, void*);
void sizeMedian(int, void*);
void rangeBilateral(int, void* );
void spaceBilateral(int, void* );


// Histogram equalization parameters
const int size = 256;
float range[] = { 0,255 };

std::string winNameGaus = "Gaussian Filter";
std::string winNameMed = "Median Filter";
std::string winNameBil = "Bilateral Filter";

// Initialization and maximum value of trackbars
int initGausSize = 11;
int initGausSigma = 5;
int maxGausSize = 50;
int maxGausSigma = 25;

int initMedSize = 11;
int maxMedSize = 50;

int initBilRange = 5;
int initBilSpace = 5;
int maxBilRange = 250;
int maxBilSpace = 25;


int main()
{
    
    // Choice of the image
    cv::Mat img;
    std::cout << "Input image:\n->";
    std::string arg;
    std::cin >> arg;
    
    std::vector<cv::Mat> imagesUsed(5);
    imagesUsed[0] = cv::imread(arg);

    while (imagesUsed[0].empty()) {
        std::cout << "Incorrect path, write the absolute path with correct extension:\n->";
        std::cin >> arg;
        imagesUsed[0] = cv::imread(arg);
    }


    imagesUsed[1] = equalizeBGRChannels(imagesUsed[0]);
    imagesUsed[2] = equalizeSingleChannel(imagesUsed[0],0);
    imagesUsed[3] = equalizeSingleChannel(imagesUsed[0],1);
    imagesUsed[4] = equalizeSingleChannel(imagesUsed[0],2);

    std::vector<std::vector<cv::Mat>> histogramChannels(5);
    for(int i =0; i<5;i++)
        histogramChannels[i] = calculateHistograms(imagesUsed[i], size, range);
	
    showImagesAndHistogram(imagesUsed,std::vector<cv::String>{"Original","RGB Equalized","H Equalized","S Equalized","V Equalized"} , histogramChannels, "window",3000 );
    

    // Creation of window and trackbar for gaussian filter 
    GaussianFilter Gaussian(imagesUsed[4], initGausSize, initGausSigma);
    sizeGaussian(initGausSize,&Gaussian);
    cv::createTrackbar("Size", winNameGaus, &initGausSize, maxGausSize, sizeGaussian, &Gaussian);
    cv::createTrackbar("Sigma", winNameGaus, &initGausSigma, maxGausSigma, sigmaGaussian, &Gaussian);


    // Creation of window and trackbar for median filter 
    MedianFilter Median(imagesUsed[4], initMedSize);
    sizeMedian(initMedSize, &Median);
    cv::createTrackbar("Size", winNameMed, &initMedSize, maxMedSize, sizeMedian, &Median);


    // Creation of window and trackbar for bilinear filter 
    BilateralFilter Bilateral(imagesUsed[4], initBilSpace * 6, initBilSpace, initBilRange);
    spaceBilateral(initBilSpace, &Bilateral);
    cv::createTrackbar("S. Space", winNameBil, &initBilSpace, maxBilSpace, spaceBilateral, &Bilateral);
    cv::createTrackbar("S. Range", winNameBil, &initBilRange, maxBilRange, rangeBilateral, &Bilateral);


    cv::waitKey();
    return 0;
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


cv::Mat equalizeSingleChannel(cv::Mat sourceImage, int channelToEqualize)
{
    cv::Mat HSVImage;
    std::vector<cv::Mat> channelsHSV(3);
	cv::cvtColor(sourceImage, HSVImage, cv::COLOR_BGR2HSV);
    cv::split(HSVImage, channelsHSV);

    cv::Mat channelsHSVTmp;
    cv::equalizeHist(channelsHSV[channelToEqualize], channelsHSVTmp);
    channelsHSV[channelToEqualize] = channelsHSVTmp;

    cv::Mat equalizedHSVImage, equalizedBGRImage;
    cv::merge(channelsHSV, equalizedHSVImage);
    cv::cvtColor(equalizedHSVImage, equalizedBGRImage, cv::COLOR_HSV2BGR);
    return equalizedBGRImage;
}

cv::Mat equalizeBGRChannels(cv::Mat sourceImage)
{
    
    std::vector<cv::Mat> channelsBGRPre(3), channelsBGRPost(3);
    cv::split(sourceImage, channelsBGRPre);
    for (int i = 0; i < 3; i++)
    {
        // Histhogram equalization of BGR channels
        cv::equalizeHist(channelsBGRPre[i], channelsBGRPost[i]);
    }
    // Merge of BGR channels equalized
    cv::Mat equalizedBGRImage;
    cv::merge(channelsBGRPost, equalizedBGRImage);
    return equalizedBGRImage; 
}

std::vector<cv::Mat> calculateHistograms(cv::Mat sourceImage, int size, float* rangeChannels)
{
    std::vector<cv::Mat> channelsPre(3),channelsPost(3);
    cv::split(sourceImage, channelsPre);
    const float* histRange = { rangeChannels };

    for (int i = 0; i < 3; i++)
    {
        // Computation of BGR histograms
        cv::calcHist(&channelsPre[i], 1, 0, cv::Mat(), channelsPost[i], 1, &size, &histRange);
    }
    return channelsPost;
}

void showImagesAndHistogram(std::vector<cv::Mat> imagesVector,std::vector<cv::String> nameVector, std::vector<std::vector<cv::Mat>> histChannels, std::string winName,int delay)
{

    // Timed change of image every between original, BGR histhogram equalized and V histhogram equalized
    int key = -1;
    int i = 0;
    cv::namedWindow(winName);
    while (key==-1)
    {

        showHistogram(histChannels[i]);
        cv::setWindowTitle(winName, nameVector[i]);
        cv::imshow(winName, imagesVector[i]);
        i = (i + 1) % imagesVector.size();
        key = cv::waitKey(delay);
    }
    cv::destroyAllWindows();
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