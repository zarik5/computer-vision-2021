#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

const int MAX_RECTS_COUNT = 20;

const int SEA_HUE = 106.f;

const float BOAT_ASPECT_RATIO = 3.f;
const float MAX_BOAT_ASPECT_RATIO = 4.f;

const float SCORE_THRESHOLD = 3;

const std::vector<float> HISTOGRAM_WEIGHTS = {2, 0.2, 1};

static cv::Mat histogram(cv::Mat input_plane, cv::Mat mask) {
    const int HISTOGRAM_BINS_COUNT = 256;
    const float HISTOGRAM_RANGE[2] = {0, HISTOGRAM_BINS_COUNT};
    const int CHANNEL = 0;

    const float *range = HISTOGRAM_RANGE;

    cv::Mat output_histogram;
    cv::calcHist(&input_plane, 1, &CHANNEL, mask, output_histogram, 1, &HISTOGRAM_BINS_COUNT,
                 &range);
    return output_histogram;
}

int main(int argc, char *argv[]) {
    auto image_path = argv[1];

    auto image = cv::imread(image_path);

    cv::Mat labels, centers;

    auto selective_search = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
    selective_search->setBaseImage(image);
    selective_search->switchToSelectiveSearchFast(1000, 1000);

    auto rects = std::vector<cv::Rect>();
    selective_search->process(rects);

    // rects = std::vector<cv::Rect>(rects.begin(), min(rects.begin() + MAX_RECTS_COUNT, rects.end()));

    cv::Mat display_image;
    image.copyTo(display_image);
    for (auto r : rects) {
        cv::rectangle(display_image, r, {0, 255, 0});
    }
    cv::imshow("hello", display_image);
    cv::waitKey();

    float max_angle = atan(BOAT_ASPECT_RATIO);
    float min_angle = atan(1 / BOAT_ASPECT_RATIO);

    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    cv::GaussianBlur(hsv_image, hsv_image, {5, 5}, 0);
    std::vector<cv::Mat> hsv_image_planes;
    cv::split(hsv_image, hsv_image_planes);

    auto image_rect = cv::Rect(0, 0, image.cols, image.rows);

    auto scores = std::vector<float>();
    for (auto r : rects) {
        float aspect_ratio = (float)r.height / (float)r.width;
        if (aspect_ratio < MAX_BOAT_ASPECT_RATIO && 1 / aspect_ratio < MAX_BOAT_ASPECT_RATIO) {
            auto center = r.tl() + (r.br() - r.tl()) / 2;

            auto candidates = std::vector<cv::RotatedRect>();
            if (aspect_ratio < BOAT_ASPECT_RATIO && 1 / aspect_ratio < BOAT_ASPECT_RATIO) {
                // approximate calculation:
                float diagonal_angle = atan(aspect_ratio);
                float rotation_progress = (diagonal_angle - min_angle) / (max_angle - min_angle);
                float angle = rotation_progress * 90;

                float a = r.width + rotation_progress * (r.width - r.height);
                float b = a / BOAT_ASPECT_RATIO;

                candidates.push_back(cv::RotatedRect(center, {a, b}, angle));
                candidates.push_back(cv::RotatedRect(center, {a, b}, -angle));
            } else {
                candidates.push_back(cv::RotatedRect(center, {(float)r.width, (float)r.height}, 0));
            }

            int context_offset = sqrt((float)r.width * r.height) / 4;
            auto context_rect =
                cv::Rect(r.x - context_offset, r.y - context_offset, r.width + context_offset * 2,
                         r.height + context_offset * 2) &
                image_rect;
            cv::Mat outer_mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);
            outer_mask(context_rect) = 255;

            // cv::imshow("hello", outer_mask);
            // cv::waitKey();

            float best_score = 0;

            for (auto c : candidates) {
                auto points = std::vector<cv::Point2f>(4);
                c.points(&points[0]);
                auto points_i =
                    std::vector<cv::Point2i>({points[0], points[1], points[2], points[3]});
                // cv::polylines(display_image, points_i, true, {0, 255, 0});

                cv::Mat boat_mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);
                cv::fillConvexPoly(boat_mask, points_i, 255, 1);

                auto context_mask = outer_mask - boat_mask;

                int channels[] = {0, 1, 2};
                int hist_sizes[] = {256, 256, 256};
                float h_range[] = {0, 256};
                float s_range[] = {0, 256};
                float v_range[] = {0, 256};
                const float *ranges[] = {h_range, s_range, v_range};

                float score = 0;
                for (int i = 0; i < 3; i++) {
                    auto boat_hist = histogram(hsv_image_planes[i], boat_mask) / cv::sum(boat_mask);
                    auto context_hist =
                        histogram(hsv_image_planes[i], context_mask) / cv::sum(context_mask);
                    score += cv::compareHist(boat_hist, context_hist, cv::HISTCMP_CORREL) *
                             HISTOGRAM_WEIGHTS[i];
                }

                cv::Mat context_hist = histogram(hsv_image_planes[0], context_mask);
                cv::Mat normalized_context_hist;
                cv::normalize(context_hist, normalized_context_hist, 100000000, 0, cv::NORM_L1);

                float weighted_sum = 0;
                for (int i = 0; i < 256; i++) {
                    weighted_sum += normalized_context_hist.at<float>(i) * i / 100000000.f;
                }
                float average = weighted_sum;

                score += 1.f / (1 + abs((SEA_HUE - average) / 3));

                // cv::imshow("hello", context_mask);
                // cv::waitKey();

                if (score > best_score) {
                    best_score = score;
                }
            }

            scores.push_back(best_score);
        } else {
            scores.push_back(0);
        }
    }

    image.copyTo(display_image);

    for (int i = 0; i < rects.size(); i++) {
        if (scores[i] > SCORE_THRESHOLD) {
            cv::rectangle(display_image, rects[i], {0, 255, 0});
        }
    }

    cv::imshow("hello", display_image);
    cv::waitKey();
}