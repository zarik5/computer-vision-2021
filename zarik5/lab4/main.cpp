#include <iostream>
#include <opencv2/opencv.hpp>

#include "circle_detection.h"
#include "lines_detection.h"

int main() {
    auto image = cv::imread("../../../data/input.png");

    auto lines_result = get_2_strongest_lines_interactive(image);

    std::cout << "========== Strongest 2 lines ==========" << std::endl
              << "Canny threshold 1: " << lines_result.canny_threshold1 << std::endl
              << "Canny threshold 2: " << lines_result.canny_threshold2 << std::endl
              << "Hough accumulator rho resolution: " << lines_result.hough_rho_resolution
              << std::endl
              << "Hough accumulator theta resolution: " << lines_result.hough_theta_resolution
              << std::endl
              << std::endl;

    auto circle_result = get_strongest_circle_interactive(image);

    std::cout << "========== Strongest circle ==========" << std::endl
              << "Canny threshold 1: " << circle_result.canny_threshold1 << std::endl
              << "Canny threshold 2: " << circle_result.canny_threshold2 << std::endl
              << "Hough inverse accumulator resolution: " << circle_result.hough_resolution
              << std::endl
              << std::endl;

    float a1 = lines_result.line1_p2.x - lines_result.line1_p1.x;
    float b1 = lines_result.line1_p2.y - lines_result.line1_p1.y;
    float a2 = lines_result.line2_p2.x - lines_result.line2_p1.x;
    float b2 = lines_result.line2_p2.y - lines_result.line2_p1.y;

    // Color pixels below both lines
    for (int x = 0; x < image.cols; x++) {
        for (int y = 0; y < image.rows; y++) {
            if (a1 * (y - lines_result.line1_p1.y) - b1 * (x - lines_result.line1_p1.x) < 0 &&
                a2 * (y - lines_result.line2_p1.y) - b2 * (x - lines_result.line2_p1.x) < 0) {
                image.at<cv::Vec3b>(y, x) = {0, 0, 255};
            }
        }
    }

    // Color circle
    cv::circle(image, circle_result.circle_center, circle_result.circle_radius, {0, 255, 0},
               cv::FILLED);

    cv::imshow("Result", image);

    cv::waitKey(0);
}
