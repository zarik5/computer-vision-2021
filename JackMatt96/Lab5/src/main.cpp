// Giacomello Mattia
// I.D. 1210988



#include <iostream>
#include <opencv2/core.hpp>
#include <vector>
#include "PanoramicImage.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;


String nameSet = "./data/dolomites";
float fov = 54;
float Ratio = 3;
int widthScreen = 1920;

int main()
{
	// Selection of all the photos inside the lab folder with bmp extension and costruction of the dataset
	vector<String> collection;
	
	glob(nameSet+"/*", collection);
	
	vector<Mat> imageSet;
	for (String name : collection)
	{
		Mat img = imread(name);
		if (!img.empty()) 
			imageSet.push_back(img);
	}

	// Creation of the class PanoramicImage and computation of the merged image
	PanoramicImage util(imageSet, fov);
	util.doStitch(Ratio);

	// Obtaining the image stitched and display of the resulting panoramic image
	Mat output = util.getResult();

	// Saving the panoramic image computed
	imwrite("Result.png", output);
	Mat outputResized;
	resize(output, outputResized, Size(widthScreen, widthScreen * output.rows / output.cols),0,0,INTER_LANCZOS4);
	
	imshow("Result", outputResized);
	
	

	waitKey();

}


 


