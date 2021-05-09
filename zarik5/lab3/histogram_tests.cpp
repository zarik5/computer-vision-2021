#include "histogram_tests.h"

const int HISTOGRAM_BINS_COUNT = 256;

// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
static void showHistogram(std::vector<cv::Mat> &hists) {
    // Min/Max computation
    double hmax[3] = {0, 0, 0};
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);

    std::string wname[3] = {"blue", "green", "red"};
    cv::Scalar colors[3] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};

    std::vector<cv::Mat> canvas(hists.size());

    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++) {
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++) {
            cv::line(canvas[i], cv::Point(j, rows),
                     cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
                     hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i], 1, 8, 0);
        }

        cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
    }
}

static cv::Mat histogram(cv::Mat input_plane) {
    const float HISTOGRAM_RANGE[2] = {0, HISTOGRAM_BINS_COUNT};
    const int CHANNEL = 0;

    const float *range = HISTOGRAM_RANGE;

    cv::Mat output_histogram;
    cv::calcHist(&input_plane, 1, &CHANNEL, cv::Mat(), output_histogram, 1, &HISTOGRAM_BINS_COUNT,
                 &range);
    return output_histogram;
}

static void calculate_and_show_histograms(std::vector<cv::Mat> image_planes_bgr) {
    std::vector<cv::Mat> histograms_rgb;
    histograms_rgb.push_back(histogram(image_planes_bgr[2]));
    histograms_rgb.push_back(histogram(image_planes_bgr[1]));
    histograms_rgb.push_back(histogram(image_planes_bgr[0]));

    showHistogram(histograms_rgb);
    cv::waitKey(0);
}

void manipulate_histograms_rgb(cv::Mat input_bgr) {
    std::vector<cv::Mat> image_planes_bgr;
    cv::split(input_bgr, image_planes_bgr);

    // Initial histograms
    calculate_and_show_histograms(image_planes_bgr);

    // Equalized image
    cv::equalizeHist(image_planes_bgr[0], image_planes_bgr[0]);
    cv::equalizeHist(image_planes_bgr[1], image_planes_bgr[1]);
    cv::equalizeHist(image_planes_bgr[2], image_planes_bgr[2]);

    calculate_and_show_histograms(image_planes_bgr);

    cv::Mat equalized_image;
    cv::merge(image_planes_bgr, equalized_image);
    cv::imshow("Equalized image (RGB channels)", equalized_image);
    cv::waitKey(0);
}

cv::Mat manipulate_histograms_hsv(cv::Mat input_bgr) {
    cv::Mat image_hsv;
    cv::cvtColor(input_bgr, image_hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> image_planes_hsv;
    cv::split(image_hsv, image_planes_hsv);

    // Equalize Value channel
    cv::equalizeHist(image_planes_hsv[2], image_planes_hsv[2]);

    cv::Mat equalized_image_hsv;
    cv::merge(image_planes_hsv, equalized_image_hsv);
    cv::Mat equalized_image_bgr;
    cv::cvtColor(equalized_image_hsv, equalized_image_bgr, cv::COLOR_HSV2BGR);
    cv::imshow("Equalized image (Value channel)", equalized_image_bgr);
    cv::waitKey(0);

    return equalized_image_bgr;
}
