#include "circle_detection.h"

#include "edge_detection.h"

// These values are found experimentally
const int CANNY_INITIAL_THRESHOLD_1 = 612;
const int CANNY_INITIAL_THRESHOLD_2 = 1296;
const double HOUGH_INITIAL_RESOLUTION = 4.5;

const int CANNY_MAX_THRESHOLD = 1500;
const double HOUGH_MAX_RESOLUTION = 5;
const double HOUGH_RESOLUTION_SCALE = 10.;
const char *const WINDOW_NAME = "Circle detection";

struct CircleOnTrackbarChangeData {
    cv::Mat input_image;
    int *canny_threshold1_ptr;
    int *canny_threshold2_ptr;
    int *hough_resolution_scaled_ptr;
    cv::Point *output_circle_center_ptr;
    float *output_circle_radius_ptr;
};

// returns: One circle paramtrized as (center_x, center_y, radius)
static cv::Vec3f find_strongest_circle(cv::Mat input, double resolution) {
    // (center_x, center_y, radius, votes)
    std::vector<cv::Vec4f> circles;
    cv::HoughCircles(input, circles, cv::HOUGH_GRADIENT, resolution, 0.1, 100, 1, 1, 20);

    cv::Vec4f strongest_circle = cv::Vec4f(0, 0, 0, 0);
    for (auto circle : circles) {
        if (circle[3] > strongest_circle[3]) {
            strongest_circle = circle;
        }
    }

    return {strongest_circle[0], strongest_circle[1], strongest_circle[2]};
}

static void on_trackbar_change(int _, void *data_vptr) {
    auto *data_ptr = (CircleOnTrackbarChangeData *)data_vptr;

    // cv::Mat gray_input;
    // cv::cvtColor(data_ptr->input_image, gray_input, cv::COLOR_BGR2GRAY);

    auto edges_image = detect_edges(data_ptr->input_image, *data_ptr->canny_threshold1_ptr,
                                    *data_ptr->canny_threshold2_ptr);

    auto circle = find_strongest_circle(
        edges_image, (double)*data_ptr->hough_resolution_scaled_ptr / HOUGH_RESOLUTION_SCALE);
    *data_ptr->output_circle_center_ptr = cv::Point(circle[0], circle[1]);
    *data_ptr->output_circle_radius_ptr = circle[2];

    auto circle_image = data_ptr->input_image.clone();
    cv::circle(circle_image, *data_ptr->output_circle_center_ptr,
               *data_ptr->output_circle_radius_ptr, {0, 0, 255}, 2);

    cv::Mat edges_gray3_image;
    cv::cvtColor(edges_image, edges_gray3_image, cv::COLOR_GRAY2BGR);

    cv::Mat image_list[2] = {edges_gray3_image, circle_image};
    cv::Mat display_image;
    cv::vconcat(image_list, 2, display_image);

    cv::imshow(WINDOW_NAME, display_image);
}

CircleResult get_strongest_circle_interactive(cv::Mat input) {
    int parameters[3] = {
        CANNY_INITIAL_THRESHOLD_1,
        CANNY_INITIAL_THRESHOLD_2,
        HOUGH_INITIAL_RESOLUTION * HOUGH_RESOLUTION_SCALE,
    };
    cv::Point circle_center;
    float circle_radius;

    CircleOnTrackbarChangeData circle_data = {
        input, &parameters[0], &parameters[1], &parameters[2], &circle_center, &circle_radius,
    };

    // Setup window
    cv::namedWindow(WINDOW_NAME);
    cv::createTrackbar("Canny threshold 1", WINDOW_NAME, circle_data.canny_threshold1_ptr,
                       CANNY_MAX_THRESHOLD, on_trackbar_change, &circle_data);
    cv::createTrackbar("Canny threshold 2", WINDOW_NAME, circle_data.canny_threshold2_ptr,
                       CANNY_MAX_THRESHOLD, on_trackbar_change, &circle_data);
    cv::createTrackbar(
        "Hough resolution / 10", WINDOW_NAME, circle_data.hough_resolution_scaled_ptr,
        HOUGH_MAX_RESOLUTION * HOUGH_RESOLUTION_SCALE, on_trackbar_change, &circle_data);

    // Calculate first test image
    on_trackbar_change(0, &circle_data);

    cv::waitKey(0);

    return {
        circle_center,
        circle_radius,
        *circle_data.canny_threshold1_ptr,
        *circle_data.canny_threshold2_ptr,
        (float)*circle_data.hough_resolution_scaled_ptr / (float)HOUGH_RESOLUTION_SCALE,
    };
}