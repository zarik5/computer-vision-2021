// Giacomello Mattia
// I.D. 1210988

#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>


void findChessboardPoints(cv::String, std::vector<std::vector<cv::Vec2f>> &, std::vector<cv::Mat> &, std::vector< cv::String >&);
void calibrateCameraChessboard(std::vector<std::vector<cv::Vec2f>> ,cv::Size , cv::Size , float ,cv::Mat& , cv::Mat& ,std::vector <double>& ,std::vector <double>& );
std::vector<cv::Vec3f> chessboard3dPoints(cv::Size , float );
cv::Mat undistortImage(cv::Mat , cv::Mat , cv::Mat );


cv::Size gridCorners(7,5);
float gridMeasure = 0.03;
cv::String chessboardDirectoryPath = "./data/";
std::string distortedImagePath = "./data/_DSC6070.jpg";

int main(){


	// Extraction from the given path of chessboard points, images and image names
	std::vector<std::vector<cv::Vec2f>> chessboardImagesPoints;
    std::vector<cv::Mat> chessboardImages;
    std::vector<cv::String >imagesName;
	findChessboardPoints(chessboardDirectoryPath, chessboardImagesPoints, chessboardImages, imagesName);
	    
	// Computation of intrinsic parameters, reprojection errors and root means squared reprojection errors throught the chessboard calibration procedure
	cv::Mat cameraMatrix, distCoeffs;
	std::vector <double> meanReprojectionErrors, meanSquaredReprojectionErrors;
	std::cout<<chessboardImagesPoints.size()<<"\n";
	calibrateCameraChessboard(chessboardImagesPoints, chessboardImages[0].size(), gridCorners, gridMeasure, cameraMatrix,  distCoeffs, meanReprojectionErrors, meanSquaredReprojectionErrors);

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

	
	// Loading and undistortion of an example image
	cv::Mat distortedImage = cv::imread(distortedImagePath);
	cv::Mat undistortedImage = undistortImage(distortedImage, cameraMatrix, distCoeffs);


	// Visualization for comparison of before and after the undistorting procedure image
	int windowHeight=1024;
	cv::resize(distortedImage, distortedImage, cv::Size(windowHeight * distortedImage.cols / distortedImage.rows, windowHeight));	
	cv::resize(undistortedImage, undistortedImage, cv::Size(windowHeight * undistortedImage.cols / undistortedImage.rows, windowHeight));
	cv::imshow("Original", distortedImage);
	cv::imshow("Undistorted", undistortedImage);
	cv::waitKey(); 

	return 0;
}


/**
* Method to retrieve from a directory the image's chessboard points to calibrate a camera 
* @param chessboardDirectory Path to the folder containing the images  
* @param chessboardImagesPoints Vector containing the points of the chessboard in the image plane for each image
* @param chessboardImages Vector containing the images containing a chessboard
* @param imagesName Vector containing the names of images containing a chessboard
*/
void findChessboardPoints(cv::String chessboardDirectory, std::vector<std::vector<cv::Vec2f>>& chessboardImagesPoints, std::vector<cv::Mat>& chessboardImages, std::vector< cv::String >& imagesName){
    
	// Estraction 
	std::vector< cv::String >imagesVectorPath;
    cv::glob(chessboardDirectory+"*",imagesVectorPath);

	std::cout<<"\nChessboard found in the images:\n";
	for(auto imagePath:imagesVectorPath)
    {
        cv::Mat img = cv::imread(imagePath);
		std::vector<cv::Vec2f> points;
		if(!cv::findChessboardCorners(img, gridCorners, points)){
			continue;
		}
		// Refining of chessboard corner and push in the vector
		cv::Mat gray;
		cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
		cv::cornerSubPix(gray, points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
		cv::drawChessboardCorners(img, gridCorners, points, true);
		chessboardImagesPoints.push_back(points);
		chessboardImages.push_back(img);
        imagesName.push_back(imagePath.substr(imagePath.find_last_of("\\")+1));
		std::cout<<imagesName.back()+"\n";
    }
}


/**
* Method to compute the intrinsic parameters of the camera from a chessboard pattern
* @param chessboardImagesPoints Vector containing the points of the chessboard in the image plane for each image
* @param imageSize Size of the images
* @param gridSize Size element containing the number of edges per row and column
* @param gridMeasure Measure in m of the edge of the chessboard squares
* @param cameraMatrix Camera matrix
* @param distCoeffs Vector containing the distortion coefficients [k1 k2 p1 p2 k3]
* @param MRE Output vector of mean reprojection error per picture
* @param MRSE Output vector of root mean squared reprojection error per picture
*/
void calibrateCameraChessboard(std::vector<std::vector<cv::Vec2f>> chessboardImagesPoints,cv::Size imageSize, cv::Size gridSize, float gridMeasure,cv::Mat &cameraMatrix, cv::Mat &distCoeffs,std::vector <double>& MRE,std::vector <double>& MRSE){  
	
	// Computation of the chessboard pattern in vector form	
	std::vector<cv::Vec3f> chessboard = chessboard3dPoints(gridSize,gridMeasure);

	// Computation of camera intrinsic and extrinsic parameters and distiortion coefficients	
	std::vector <cv::Mat> rvecs, tvecs;
	cv::calibrateCamera(std::vector<std::vector<cv::Vec3f>> (chessboardImagesPoints.size(),chessboard), chessboardImagesPoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    // Computation of mean reprojection errors and root mean squared reprojection errors
	MRE = std::vector <double> (chessboardImagesPoints.size());
    MRSE = std::vector <double> (chessboardImagesPoints.size());

	for (size_t i = 0; i < chessboardImagesPoints.size(); i++)
	{
		std::vector <cv::Vec2f> chessboardReprojected;
		cv::projectPoints(chessboard, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, chessboardReprojected);
		
        double error = 0;
		for (size_t j = 0; j < chessboardReprojected.size(); j++)
		{
			error += cv::norm(chessboardReprojected[j]- chessboardImagesPoints[i][j]);
		}
        MRE[i] = error / chessboardImagesPoints[i].size();
                
        error = 0;
		for (size_t j = 0; j < chessboardReprojected.size(); j++)
		{
			error += pow(cv::norm(chessboardReprojected[j]- chessboardImagesPoints[i][j]),2);
		}
        MRSE[i] = sqrt(error) / chessboardImagesPoints[i].size();      
	}
}

/**
* Method to build the 3D reference chessboard with Z=0. The top left corner is placed to (0,0,0)
* @param gridSize Size element containing the number of edges per row and column
* @param gridMeasure Measure in m of the edge of the chessboard squares
* @return vector containing the 3D world points of the chessboard corners
*/
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


/**
* Method to undistort an image
* @param distortedImage Distorted image
* @param cameraMatrix Camera matrix
* @param distCoeffs Vector containing the distortion coefficients [k1 k2 p1 p2 k3]
* @return Undistorted image
*/
cv::Mat undistortImage(cv::Mat distortedImage, cv::Mat cameraMatrix, cv::Mat distCoeffs){
	// Remapping of distorted images
	cv::Mat mapx, mapy;
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, distortedImage.size(), CV_32FC1, mapx, mapy);
	
	cv::Mat undistortedImage;
	cv::remap(distortedImage, undistortedImage, mapx,mapy, cv::INTER_LANCZOS4);
	return undistortedImage;
}