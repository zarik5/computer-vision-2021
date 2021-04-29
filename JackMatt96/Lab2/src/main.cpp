// Giacomello Mattia
// I.D. 1210988

#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>


void findChessboardPoints(std::vector< cv::String >, std::vector<std::vector<cv::Vec2f>> , std::vector<cv::Mat> , std::vector< cv::String >);
void calibrateCameraChessboard(std::vector<std::vector<cv::Vec2f>>,cv::Size, cv::Size , float ,cv::Mat , cv::Mat ,std::vector <double> ,std::vector <double> );
std::vector<cv::Vec3f> chessboard3dPoints(cv::Size , float );
cv::Mat undistortImage(cv::Mat , cv::Mat , cv::Mat );

const int ESCKey = 27;
const int windowHeight = 1024;
const std::string orig_name = "Original photo. ESC to close";
const std::string und_name = "Undistorted photo. ESC to close";

cv::Size gridSize(7,5);
float gridMeasure = 0.02;
cv::String chessboardDirectory = "./data/",
pattern = "*.jpg";


int main(){

    std::vector< cv::String >imagesVectorPath;
    cv::utils::fs::glob(chessboardDirectory,pattern,imagesVectorPath);

	std::vector<std::vector<cv::Vec2f>> chessboardImagesPoints;
    std::vector<cv::Mat> chessboardImages;
    std::vector<cv::String >imagesName;
	
	findChessboardPoints(imagesVectorPath, chessboardImagesPoints, chessboardImages, imagesName);
	
    
	cv::Mat cameraMatrix, distCoeffs;
	std::vector <double> meanReprojectionErrors, meanSquaredReprojectionErrors;

	calibrateCameraChessboard(chessboardImagesPoints, chessboardImages[0].size(),gridSize, gridMeasure, cameraMatrix,  distCoeffs, meanReprojectionErrors, meanSquaredReprojectionErrors);

	int minIdx, maxIdx, minSIdx, maxSIdx;
	cv::minMaxIdx(meanReprojectionErrors,NULL,NULL,&minIdx,&maxIdx);
	cv::minMaxIdx(meanSquaredReprojectionErrors,NULL,NULL,&minSIdx,&maxSIdx);

    // Print to terminal of values obtained
	std::cout << "\nIntrinsic Parameters: \n" << cameraMatrix
		<< "\n\nLens coefficients: \n[k1 k2 p1 p2 k3] = " << distCoeffs
		<< "\nBest image: " << imagesName[minIdx] <<" with mean reprojection error = "<< meanReprojectionErrors[minIdx] 
		<< "\nWorst image: " << imagesName[maxIdx] << " with mean reprojection error = " << meanReprojectionErrors[maxIdx]
		<< "\nBest image: " << imagesName[minSIdx] <<" with mean squared reprojection error = "<< meanSquaredReprojectionErrors[minSIdx] 
		<< "\nWorst image: " << imagesName[maxSIdx] << " with mean squared reprojection error = " << meanSquaredReprojectionErrors[maxSIdx] << "\n";
    

    // Choice of distorted image to remap

	std::string distortedImagePath;
	std::cout << "Path of distorted image:\n->";
	std::cin >> distortedImagePath;
    
	cv::Mat distortedImage = cv::imread(distortedImagePath);
	while (distortedImage.empty()) {
		std::cout << "Incorrect path, retry:\n->";
		std::cin >> distortedImagePath;
		distortedImage = cv::imread(distortedImagePath);
	}
	
	cv::Mat undistortedImage = undistortImage(distortedImage, cameraMatrix, distCoeffs);

	cv::resize(undistortedImage, undistortedImage, cv::Size(windowHeight * undistortedImage.cols / undistortedImage.rows, windowHeight));
	cv::resize(distortedImage, distortedImage, cv::Size(windowHeight * distortedImage.cols / distortedImage.rows, windowHeight));	

	// Visualization of distorted and undistorted images
	cv::imshow(orig_name, distortedImage);
	cv::imshow(und_name, undistortedImage);
	while (cv::waitKey() != ESCKey); // Esc Key to close
	return 0;
}



void findChessboardPoints(std::vector< cv::String >imagesVectorPath, std::vector<std::vector<cv::Vec2f>> chessboardImagesPoints, std::vector<cv::Mat> chessboardImages, std::vector< cv::String >imagesName){
	for(auto imagePath:imagesVectorPath)
    {
        cv::Mat img = cv::imread(imagePath);
		std::vector<cv::Vec2f> points;
		if (!img.empty() && cv::findChessboardCorners(img, gridSize, points)) {
			// Refining of chessboard corner and push in the vector
			cv::Mat gray;
			cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
			cv::cornerSubPix(gray, points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
			cv::drawChessboardCorners(img, gridSize, points, true);
			chessboardImagesPoints.push_back(points);
			chessboardImages.push_back(img);
            imagesName.push_back(imagePath.substr(imagePath.find_last_of("/")));
		}
    }
}


void calibrateCameraChessboard(std::vector<std::vector<cv::Vec2f>> chessboardImagesPoints,cv::Size imageSize, cv::Size gridSize, float gridMeasure,cv::Mat cameraMatrix, cv::Mat distCoeffs,std::vector <double> meanReprojectionErrors,std::vector <double> meanSquaredReprojectionErrors){
	
	std::vector<cv::Vec3f> chessboard = chessboard3dPoints(gridSize,gridMeasure);
	// Computation of camera intrinsic and extrinsic parameters and lens coefficients
	
	std::vector <cv::Mat> rvecs, tvecs;
	cv::calibrateCamera(std::vector<std::vector<cv::Vec3f>> (chessboardImagesPoints.size(),chessboard), chessboardImagesPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    // Computation of single root mean squared reprojection errors
	meanReprojectionErrors = std::vector <double> (chessboardImagesPoints.size());
    meanSquaredReprojectionErrors = std::vector <double> (chessboardImagesPoints.size());

	for (size_t i = 0; i < chessboardImagesPoints.size(); i++)
	{
		std::vector <cv::Vec2f> chessboardReprojected;
		cv::projectPoints(chessboard, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, chessboardReprojected);
		
        double error = 0;
		for (size_t j = 0; j < chessboardReprojected.size(); j++)
		{
			error += cv::norm(chessboardReprojected[j]- chessboardImagesPoints[i][j]);
		}
        meanReprojectionErrors[i] = error / chessboardImagesPoints[i].size();
                
        error = 0;
		for (size_t j = 0; j < chessboardReprojected.size(); j++)
		{
			error += pow(cv::norm(chessboardReprojected[j]- chessboardImagesPoints[i][j]),2);
		}
        meanSquaredReprojectionErrors[i] = error / chessboardImagesPoints[i].size();        
	}
}


std::vector<cv::Vec3f> chessboard3dPoints(cv::Size gridSize, float gridMeasure){
	// Creation of vector of virtual chessboard
	std::vector<cv::Vec3f> chessboard;
	for (size_t i = 0; i < gridSize.height; i++)
	{
		for (size_t j = 0; j < gridSize.width; j++)
		{
			chessboard.push_back(cv::Vec3f(j * gridMeasure, i * gridMeasure, 0));
		}
	}
	return chessboard;
}


cv::Mat undistortImage(cv::Mat distortedImage, cv::Mat cameraMatrix, cv::Mat distCoeffs){
	// Remapping of distorted images
	cv::Mat map1, map2;
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, distortedImage.size(), CV_32FC1, map1, map2);
	
	cv::Mat undistortedImage;
	cv::remap(distortedImage, undistortedImage, map1,map2, cv::INTER_LANCZOS4);
	return undistortedImage;
}