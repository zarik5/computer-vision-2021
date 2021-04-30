#include <iostream>

#include <opencv2/opencv.hpp>

const int CHESSBOARD_CORNERS_X = 6;
const int CHESSBOARD_CORNERS_Y = 5;
const float CHESSBOARD_SQUARE_SIDE_M = 0.11;

const int CORNERS_REFINEMENT_TERM_COUNT = 30;
const double CORNERS_REFINEMENT_TERM_EPS = 0.0001;
const auto IMAGE_SIZE = cv::Size(1928, 1448);

int main() {
    std::vector<std::string> chessboard_file_paths;

    cv::glob("../../../data/chessboard/*", chessboard_file_paths);

    // coordinates of corners obtained from input images
    std::vector<std::vector<cv::Point2f>> valid_captured_corners;
    std::vector<std::string> valid_file_paths;
    for (auto file_path : chessboard_file_paths) {
        std::cout << "Processing image: " << file_path << std::endl;

        auto image = cv::imread(file_path);

        std::vector<cv::Point2f> corners;
        if (cv::findChessboardCorners(image, {5, 6}, corners)) {
            cv::Mat gray_image;
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

            auto term =
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                 CORNERS_REFINEMENT_TERM_COUNT, CORNERS_REFINEMENT_TERM_EPS);
            cv::cornerSubPix(image, corners, {5, 5}, {-1, -1}, term);

            valid_captured_corners.push_back(corners);
            valid_file_paths.push_back(file_path);
        } else {
            std::cout << "Discarded!" << std::endl;
        }
    }

    // coordinates of corners in world (chessboard) space, one set per input image
    std::vector<std::vector<cv::Point2f>> template_corners;
    {
        std::vector<cv::Point2f> corners;
        for (int x = 0; x < CHESSBOARD_CORNERS_X; x++) {
            for (int y = 0; y < CHESSBOARD_CORNERS_Y; y++) {
                corners.push_back({x * CHESSBOARD_SQUARE_SIDE_M, y * CHESSBOARD_SQUARE_SIDE_M});
            }
        }
        template_corners = std::vector(valid_captured_corners.size(), corners);
    }

    std::cout << "Calibrating camera..." << std::endl;
    cv::Mat camera_matrix;
    cv::Mat distortion_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    cv::calibrateCamera(template_corners, valid_captured_corners, IMAGE_SIZE, camera_matrix,
                        distortion_coeffs, rvecs, tvecs);
    std::cout << "Intrinsic Parameters [3x3]: " << camera_matrix << std::endl
              << "Lens coefficients [k1 k2 p1 p2 k3]: " << distortion_coeffs << std::endl;

    auto reprojection_errors = std::vector<double>();
    // int min_
}
