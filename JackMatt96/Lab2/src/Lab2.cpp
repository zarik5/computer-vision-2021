// Giacomello Mattia
// I.D. 1210988


#include <stdlib.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

const int ESCKey = 27;
const int windowHeight = 1024;
const std::string orig_name = "Original photo. ESC to close";
const std::string und_name = "Undistorted photo. ESC to close";

int main(){
	// User interface to ask user data
	std::vector <std::string> chessNames;
	std::string images_folder;
	float gridSize;
	int gridCols;
	int gridRows;

	std::cout << "Folder containing calibration images:\n->";
	std::cin >> images_folder;
	cv::glob(images_folder + "/*", chessNames);
	std::cout << "Number of columns inner intersections:\n->";
	std::cin >> gridCols;
	std::cout << "Number of rows inner intersections:\n->";
	std::cin >> gridRows;
	std::cout << "Grid size in m:\n->";
	std::cin >> gridSize;

	std::vector <cv::Mat> imagesChess;
	std::vector <std::vector<cv::Vec2f>> image_points;

	std::cout << "\nSearching for images with chessboard of "+ std::to_string(gridCols)+"x"+ std::to_string(gridRows)+ " of size "+ std::to_string(gridSize)+"m in the folder "+ images_folder+
		"\nChessboard found in: \n";
	// Search for chessboard corners in the files in the path defined
	for (size_t i = 0; i < chessNames.size(); i ++ )
	{
		cv::Mat img = cv::imread(chessNames[i]);
		if (img.empty()) {
			chessNames.erase(chessNames.begin() + i);
			i--;
			continue;
		}
		std::vector<cv::Vec2f> points;
		if (cv::findChessboardCorners(img, cv::Size(gridCols, gridRows), points)) {
			// Refining of chessboard corner and push in the vector
			cv::Mat gray;
			cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
			cv::cornerSubPix(gray, points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
			cv::drawChessboardCorners(img, cv::Size(gridCols, gridRows), points, true);
			image_points.push_back(points);
			imagesChess.push_back(img);
			std::cout << chessNames[i]+"\n";
		}
		else {
			chessNames.erase(chessNames.begin()+i);
			i--;
		}
	}
	

	// Creation of vector of virtual chessboard
	std::vector<cv::Vec3f> chessTemp;
	for (size_t i = 0; i < gridRows; i++)
	{
		for (size_t j = 0; j < gridCols; j++)
		{
			chessTemp.push_back(cv::Vec3f(j * gridSize, i * gridSize, 0));
		}
	}
	std::vector<std::vector<cv::Vec3f>> chessboard(imagesChess.size(),chessTemp);


	// Computation of camera intrinsic and extrinsic parameters and lens coefficients
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
	std::vector <cv::Mat> rvecs;
	std::vector <cv::Mat> tvecs;
	cv::calibrateCamera(chessboard, image_points, imagesChess[0].size(), cameraMatrix, distCoeffs, rvecs, tvecs);


	// Computation of single root mean squared reprojection errors
	std::vector <double> reprojectionErrors(imagesChess.size());
	int minIdx = 0, maxIdx = 0;
	double RMS = 0;

	for (size_t i = 0; i < imagesChess.size(); i++)
	{
		std::vector <cv::Vec2f> chessReprojected;
		cv::projectPoints(chessboard[0], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, chessReprojected);
		double error = 0;
		for (size_t j = 0; j < chessReprojected.size(); j++)
		{
			error += cv::pow(cv::norm(chessReprojected[j]- image_points[i][j]),2);

		}
		RMS += error;
		reprojectionErrors[i] = cv::sqrt(error / chessboard[0].size());
		if (reprojectionErrors[i]> reprojectionErrors[maxIdx])
		{
			maxIdx = i;
		}
		else if (reprojectionErrors[i] < reprojectionErrors[minIdx])
		{
			minIdx = i;
		}
	}

	RMS = cv::sqrt(RMS / (chessboard[0].size() * chessboard.size()));

	// Print to terminal of values obtained
	std::cout << "\nReprojection Error: " << RMS
		<< "\nIntrinsic Parameters: \n" << cameraMatrix
		<< "\n\nLens coefficients: \n[k1 k2 p1 p2 k3] = " << distCoeffs
		<< "\nBest image: " << chessNames[minIdx] <<" with RMS = "<< reprojectionErrors[minIdx] 
		<< "\nWorst image: " << chessNames[maxIdx] << " with RMS = " << reprojectionErrors[maxIdx] << "\n";
	
	// Choice of distorted image to remap
	std::string distortedName;
	std::cout << "Folder containing calibration images:\n->";
	std::cin >> distortedName;

	cv::Mat imagesDistorted = cv::imread(distortedName);
	while (imagesDistorted.empty()) {
		std::cout << "Incorrect path, retry:\n->";
		std::cin >> distortedName;
		imagesDistorted = cv::imread(distortedName);
	}
	
	// Remapping of distorted images
	cv::Mat map1, map2;
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, imagesDistorted.size(), CV_32FC1, map1, map2);
	
	cv::Mat imgUndist;
	cv::remap(imagesDistorted, imgUndist, map1,map2, cv::INTER_LANCZOS4);
	cv::resize(imgUndist, imgUndist, cv::Size(windowHeight * imgUndist.cols / imgUndist.rows, windowHeight));
	cv::resize(imagesDistorted, imagesDistorted, cv::Size(windowHeight * imagesDistorted.cols / imagesDistorted.rows, windowHeight));
	

	// Visualization of distorted and undistorted images
	cv::imshow(orig_name, imagesDistorted);
	cv::imshow(und_name, imgUndist);
	while (cv::waitKey() != ESCKey); // Esc Key to close
	return 0;
}