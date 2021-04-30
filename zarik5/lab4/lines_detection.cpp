#include "lines_detection.h"

#include "edge_detection.h"

// These values are found experimentally
const int CANNY_INITIAL_THRESHOLD_1 = 610;
const int CANNY_INITIAL_THRESHOLD_2 = 794;
const double HOUGH_INITIAL_RHO_RESOLUTION = 0.4;
const double HOUGH_INITIAL_THETA_RESOLUTION = 0.11;

const int CANNY_MAX_THRESHOLD = 1000;
const double HOUGH_MAX_RHO_RESOLUTION = 1;
const double HOUGH_MAX_THETA_RESOLUTION = 0.5;
const double HOUGH_RESOLUTION_SCALE = 100.;
const char *const WINDOW_NAME = "Lines detection";

struct OnTrackbarChangeData {
    cv::Mat input_image;
    int *canny_threshold1_ptr;
    int *canny_threshold2_ptr;
    int *hough_rho_resolution_scaled_ptr;
    int *hough_theta_resolution_scaled_ptr;
    cv::Point *output_line1_p1_ptr;
    cv::Point *output_line1_p2_ptr;
    cv::Point *output_line2_p1_ptr;
    cv::Point *output_line2_p2_ptr;
};

// returns: 2 lines are parametrized as (rho, theta)
static std::array<cv::Vec2f, 2> find_2_strongest_lines(cv::Mat input, double rho_resolution,
                                                       double theta_resolution) {
    // (rho, theta, votes)
    std::vector<cv::Vec3f> lines;
    cv::HoughLines(input, lines, (double)rho_resolution, (double)theta_resolution, 0);

    // this could be simplified by sorting the vector first, but the following code should be more
    // efficient
    cv::Vec3f strongest_line = cv::Vec3f(0, 0, 0);
    cv::Vec3f second_strongest_line = cv::Vec3f(0, 0, 0);
    for (auto line : lines) {
        if (line[2] > strongest_line[2]) {
            strongest_line = line;
        } else if (line[2] > second_strongest_line[2]) {
            second_strongest_line = line;
        }
    }

    return {cv::Vec2f(strongest_line[0], strongest_line[1]),
            cv::Vec2f(second_strongest_line[0], second_strongest_line[1])};
}

// returns the two extremes of the segment
static std::array<cv::Point, 2> line_polar_to_cartesian(float rho, float theta,
                                                        float segment_length) {
    float a = cos(theta);
    float b = sin(theta);
    float px = rho * a;
    float py = rho * b;

    // using similar triangles:
    // px : rho = rho : x_cross
    float x_cross = rho * rho / cv::max(px, 0.000001f);
    // py : rho = rho : y_cross
    float y_cross = rho * rho / cv::max(py, 0.000001f);

    return {cv::Point(px - segment_length * -b / 2, py - (float)segment_length * a / 2),
            cv::Point(px + segment_length * -b / 2, py + (float)segment_length * a / 2)};
}

static void on_trackbar_change(int _, void *data_vptr) {
    auto *data_ptr = (OnTrackbarChangeData *)data_vptr;

    auto edges_image = detect_edges(data_ptr->input_image, *data_ptr->canny_threshold1_ptr,
                                    *data_ptr->canny_threshold2_ptr);

    auto [line1_polar, line2_polar] = find_2_strongest_lines(
        edges_image, (double)*data_ptr->hough_rho_resolution_scaled_ptr / HOUGH_RESOLUTION_SCALE,
        (double)*data_ptr->hough_theta_resolution_scaled_ptr / HOUGH_RESOLUTION_SCALE);

    auto greater_dimension = cv::max(data_ptr->input_image.rows, data_ptr->input_image.cols);
    auto [line1_p1, line1_p2] =
        line_polar_to_cartesian(line1_polar[0], line1_polar[1], greater_dimension * 2);
    auto [line2_p1, line2_p2] =
        line_polar_to_cartesian(line2_polar[0], line2_polar[1], greater_dimension * 2);
    *data_ptr->output_line1_p1_ptr = line1_p1;
    *data_ptr->output_line1_p2_ptr = line1_p2;
    *data_ptr->output_line2_p1_ptr = line2_p1;
    *data_ptr->output_line2_p2_ptr = line2_p2;

    auto lines_image = data_ptr->input_image.clone();
    cv::line(lines_image, line1_p1, line1_p2, {0, 0, 255}, 2);
    cv::line(lines_image, line2_p1, line2_p2, {0, 0, 255}, 2);

    cv::Mat edges_gray3_image;
    cv::cvtColor(edges_image, edges_gray3_image, cv::COLOR_GRAY2BGR);

    cv::Mat image_list[2] = {edges_gray3_image, lines_image};
    cv::Mat display_image;
    cv::vconcat(image_list, 2, display_image);

    cv::imshow(WINDOW_NAME, display_image);
}

LinesResult get_2_strongest_lines_interactive(cv::Mat input) {
    int parameters[4] = {
        CANNY_INITIAL_THRESHOLD_1,
        CANNY_INITIAL_THRESHOLD_2,
        HOUGH_INITIAL_RHO_RESOLUTION * HOUGH_RESOLUTION_SCALE,
        HOUGH_INITIAL_THETA_RESOLUTION * HOUGH_RESOLUTION_SCALE,
    };
    cv::Point lines_points[4];

    OnTrackbarChangeData lines_data = {
        input,
        &parameters[0],
        &parameters[1],
        &parameters[2],
        &parameters[3],
        &lines_points[0],
        &lines_points[1],
        &lines_points[2],
        &lines_points[3],
    };

    // Setup window
    cv::namedWindow(WINDOW_NAME);
    cv::createTrackbar("Canny threshold 1", WINDOW_NAME, lines_data.canny_threshold1_ptr,
                       CANNY_MAX_THRESHOLD, on_trackbar_change, &lines_data);
    cv::createTrackbar("Canny threshold 2", WINDOW_NAME, lines_data.canny_threshold2_ptr,
                       CANNY_MAX_THRESHOLD, on_trackbar_change, &lines_data);
    cv::createTrackbar(
        "Hough rho resolution / 100", WINDOW_NAME, lines_data.hough_rho_resolution_scaled_ptr,
        HOUGH_MAX_RHO_RESOLUTION * HOUGH_RESOLUTION_SCALE, on_trackbar_change, &lines_data);
    cv::createTrackbar(
        "Hough theta resolution / 100", WINDOW_NAME, lines_data.hough_theta_resolution_scaled_ptr,
        HOUGH_MAX_THETA_RESOLUTION * HOUGH_RESOLUTION_SCALE, on_trackbar_change, &lines_data);

    // Calculate first test image before moving the trackbar knobs
    on_trackbar_change(0, &lines_data);

    cv::waitKey(0);

    return {
        *lines_data.output_line1_p1_ptr,
        *lines_data.output_line1_p2_ptr,
        *lines_data.output_line2_p1_ptr,
        *lines_data.output_line2_p2_ptr,
        *lines_data.canny_threshold1_ptr,
        *lines_data.canny_threshold2_ptr,
        (float)*lines_data.hough_rho_resolution_scaled_ptr / (float)HOUGH_RESOLUTION_SCALE,
        (float)*lines_data.hough_theta_resolution_scaled_ptr / (float)HOUGH_RESOLUTION_SCALE,
    };
}