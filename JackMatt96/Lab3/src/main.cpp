// Giacomello Mattia
// I.D. 1210988


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "filter.h"

// Definition of auxiliary functions
void showHistogram(std::vector<cv::Mat>&);
void sizeGaussian(int,void*);
void sigmaGaussian(int, void*);
void sizeMedian(int, void*);
void rangeBilateral(int , void* );
void spaceBilateral(int , void* );

// Histogram equalization parameters
const int size = 256;
const float range[] = { 0,255 };
const float* rangeChans = { range };
const int HSVSelector = 2; // Selection of Intensity channel

// Visualization parameters
std::string winName = "Window";
const int winHeight = 1024;
std::string winNameGaus = "Gaussian Filter";
std::string winNameMed = "Median Filter";
std::string winNameBil = "Billateral Filter";
const int delay = 3000; // between images in ms

// Initialization and maximum value of trackbars
int initGausSize = 11;
int initGausSigma = 5;
int maxGausSize = 255;
int maxGausSigma = 50;

int initMedSize = 11;
int maxMedSize = 255;

int initBilRange = 5;
int initBilSpace = 5;
int maxBilRange = 250;
int maxBilSpace = 50;

// Auxiliary structure
struct ImageStruct
{
    int param1;
    int param2;
    cv::Mat src_img, res_img;
};



int main()
{
    // Choice of the image
    cv::Mat img;
    std::cout << "Input image:\n->";
    std::string arg;
    std::cin >> arg;
    img = cv::imread(arg);

    while (img.empty()) {
        std::cout << "Incorrect path, write the absolute path with correct extension:\n->";
        std::cin >> arg;
        img = cv::imread(arg);
    }

    cv::resize(img, img, cv::Size(winHeight * img.cols / img.rows, winHeight),.0,.0,cv::INTER_LANCZOS4);
    cv::Mat imgHSV, imgBGRPost;
	cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> channelsBGRPre(3);
	std::vector<cv::Mat> channelsBGRPost(3);
    std::vector<cv::Mat> channelsHSV(3);

	std::vector<cv::Mat> histChannelsBGRPre(3);
	std::vector<cv::Mat> histChannelsBGRPost(3);
    std::vector<cv::Mat> histChannelsHSVPost(3);
    cv::Mat channelsHSVTmp;


    // V histhogram equalization
    cv::split(imgHSV, channelsHSV);
    cv::equalizeHist(channelsHSV[HSVSelector], channelsHSVTmp);
    channelsHSV[HSVSelector] = channelsHSVTmp;
    cv::merge(channelsHSV, imgHSV);
    cv::cvtColor(imgHSV, imgHSV, cv::COLOR_HSV2BGR);

    // Split into BGR channels
    cv::split(imgHSV, channelsHSV);
    cv::split(img, channelsBGRPre);
    for (int i = 0; i < 3; i++)
    {
        // Histhogram equalization of BGR channels
        cv::equalizeHist(channelsBGRPre[i], channelsBGRPost[i]);
        // Computation of histograms for original, BRG equalized and V channel equalized
        cv::calcHist(&channelsBGRPre[i], 1, 0, cv::Mat(), histChannelsBGRPre[i], 1, &size, &rangeChans);
        cv::calcHist(&channelsBGRPost[i], 1, 0, cv::Mat(), histChannelsBGRPost[i], 1, &size, &rangeChans);
        cv::calcHist(&channelsHSV[i], 1, 0, cv::Mat(), histChannelsHSVPost[i], 1, &size, &rangeChans);
    }
    // Merge of BGR channels equalized
    cv::merge(channelsBGRPost, imgBGRPost);

    // Timed change of image every between original, BGR histhogram equalized and V histhogram equalized
    int key = -1;
    int i = 0;
    cv::namedWindow(winName);
    while (key==-1)
    {
        switch (i)
        {
        case 0:
            showHistogram(histChannelsBGRPre);
            cv::setWindowTitle(winName, "Original image");
            cv::imshow(winName, img);
        break;
        case 1:
            showHistogram(histChannelsBGRPost);
            cv::setWindowTitle(winName, "Image BGR equalized");
            cv::imshow(winName, imgBGRPost);
        break;
        case 2:
            showHistogram(histChannelsHSVPost);
            cv::setWindowTitle(winName, "Image V equalized");
            cv::imshow(winName, imgHSV);
        break;
        default:
            i = 0;
        }
        i = (i + 1) % 3;
        key = cv::waitKey(delay);
    }
    cv::destroyAllWindows();

    // Inizialization of filters
    ImageStruct GaussianStruct = { initGausSize, initGausSigma,imgHSV };
    ImageStruct MedianStruct = { initMedSize, 0 ,imgHSV };
    ImageStruct BilateralStruct = { initBilRange, initBilSpace ,imgHSV };

    // Creation of window and trackbar for gaussian filter 
    cv::namedWindow(winNameGaus);
    sizeGaussian(0, &MedianStruct);
    cv::createTrackbar("Size", winNameGaus, &GaussianStruct.param1, maxGausSize, sizeGaussian, &GaussianStruct);
    cv::createTrackbar("Sigma", winNameGaus, &GaussianStruct.param2, maxGausSigma, sigmaGaussian, &GaussianStruct);


    // Creation of window and trackbar for median filter 
    cv::namedWindow(winNameMed);
    sizeMedian(0, &MedianStruct);
    cv::createTrackbar("Size", winNameMed, &MedianStruct.param1, maxMedSize, sizeMedian, &MedianStruct);


    // Creation of window and trackbar for bilinear filter 
    cv::namedWindow(winNameBil);
    spaceBilateral(0, &BilateralStruct);
    cv::createTrackbar("S. Space", winNameBil, &BilateralStruct.param1, maxBilSpace, spaceBilateral, &BilateralStruct);
    cv::createTrackbar("S. Range", winNameBil, &BilateralStruct.param2, maxBilRange, rangeBilateral, &BilateralStruct);

    cv::waitKey();
    return 0;
}


// Callbacks for change of sliders: filtering and show filtered image.
void sizeGaussian(int , void* GaussianStruct) {
    ImageStruct* GausStruct = (ImageStruct*)GaussianStruct;
    GaussianFilter Gaussian(GausStruct->src_img, GausStruct->param1, GausStruct->param2);
    Gaussian.doFilter();
    GausStruct->res_img = Gaussian.getResult();
    cv::imshow(winNameGaus, GausStruct->res_img);
}

void sigmaGaussian(int , void* GaussianStruct) {
    ImageStruct* GausStruct = (ImageStruct*)GaussianStruct;
    GaussianFilter Gaussian(GausStruct->src_img, GausStruct->param1, GausStruct->param2);
    Gaussian.doFilter();
    GausStruct->res_img = Gaussian.getResult();
    cv::imshow(winNameGaus, GausStruct->res_img);
}


void sizeMedian(int , void* MedianStruct) {
    ImageStruct* MedStruct = (ImageStruct*)MedianStruct;
    MedianFilter Median(MedStruct->src_img, MedStruct->param1);
    Median.doFilter();
    MedStruct->res_img = Median.getResult();
    cv::imshow(winNameMed, MedStruct->res_img);
}

void spaceBilateral(int, void* BilateralStruct) {
    ImageStruct* BilStruct = (ImageStruct*)BilateralStruct;
    BilateralFilter Bilateral(BilStruct->src_img, BilStruct->param1 * 6, BilStruct->param2, BilStruct->param1);
    Bilateral.doFilter();
    BilStruct->res_img = Bilateral.getResult();
    cv::imshow(winNameBil, BilStruct->res_img);
}

void rangeBilateral(int , void* BilateralStruct) {
    ImageStruct* BilStruct = (ImageStruct*)BilateralStruct;
    BilateralFilter Bilateral(BilStruct->src_img, BilStruct->param1 * 6, BilStruct->param2, BilStruct->param1);
    Bilateral.doFilter();
    BilStruct->res_img = Bilateral.getResult();
    cv::imshow(winNameBil, BilStruct->res_img);
}





// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat>& hists)
{
    // Min/Max computation
    double hmax[3] = { 0,0,0 };
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

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
        {
            cv::line(
                canvas[i],
                cv::Point(j, rows),
                cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
                hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
                1, 8, 0
            );
        }

        cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
    }
}