#include "panoramic_image.h"
#include "panoramic_utils.h"

static std::vector<cv::Mat> extract_features(cv::Mat &image) {}

PanoramicImage::PanoramicImage(std::vector<cv::Mat> input_images) : input_images(input_images) {}

cv::Mat PanoramicImage::stitch(float half_fov_deg, float distance_ratio) {
    std::vector<cv::Mat> projected_images;
    for (auto &image : this->input_images) {
        auto proj = PanoramicUtils::cylindricalProj(image, half_fov_deg);

        // Remove 1 pixel border. It seems cylindricalProj() has a bug where those pixels
        // are shifted or distorted
        projected_images.push_back(proj(cv::Range(1, proj.rows - 1), cv::Range(1, proj.cols - 1)));
    }

    auto feature_detector = cv::SIFT::create();
    auto keypoints = std::vector<std::vector<cv::KeyPoint>>(projected_images.size());
    auto descriptors = std::vector<cv::Mat>(projected_images.size());
    for (int i = 0; i < projected_images.size(); i++) {
        feature_detector->detectAndCompute(projected_images[i], cv::Mat(), keypoints[i],
                                           descriptors[i]);
    }

    // Cumulative translations. The element (0, 0) is already inserted: the first image position is
    // the reference frame
    std::vector<cv::Point> translations = {cv::Point(0, 0)};
    auto matcher = cv::BFMatcher::create(cv::NORM_L2);
    for (int i = 1; i < projected_images.size(); i++) {
        std::vector<cv::DMatch> pair_matches;
        matcher->match(descriptors[i], descriptors[i - 1], pair_matches);

        float min_distance = FLT_MAX;
        for (auto &match : pair_matches) {
            if (min_distance > match.distance) {
                min_distance = match.distance;
            }
        }

        std::vector<cv::DMatch> filtered_matches;
        for (auto &match : pair_matches) {
            if (match.distance < distance_ratio * min_distance) {
                filtered_matches.push_back(match);
            }
        }

        std::vector<cv::Point2f> keypoints1;
        std::vector<cv::Point2f> keypoints2;
        for (auto &match : filtered_matches) {
            keypoints1.push_back(keypoints[i - 1][match.trainIdx].pt);
            keypoints2.push_back(keypoints[i][match.queryIdx].pt);
        }

        cv::Mat inliers_mask;
        cv::findHomography(keypoints1, keypoints2, inliers_mask, cv::RANSAC);

        int inliers_count = 0;
        auto inlier_translation_sum = cv::Point(0, 0);
        for (int j = 0; j < inliers_mask.rows; j++) {
            if (inliers_mask.at<uint8_t>(j, 0)) {
                inliers_count++;
                inlier_translation_sum += cv::Point(keypoints1[j] - keypoints2[j]);
            }
        }

        translations.push_back(translations[i - 1] + inlier_translation_sum / inliers_count);
    }

    // Find a suitable ROI for the final panorama. Find the left and right edge positions that
    // results in the minimum image width to contain all input images. Find the top and bottom edge
    // positions that result in the maximum image height that does not contain black borders.
    int parorama_x_left = INT_MAX;
    int parorama_x_right = -INT_MAX;
    int parorama_y_top = -INT_MAX;
    int parorama_y_bottom = INT_MAX;
    for (int i = 0; i < projected_images.size(); i++) {
        if (parorama_x_left > translations[i].x) {
            parorama_x_left = translations[i].x;
        }
        if (parorama_x_right < translations[i].x + projected_images[i].cols) {
            parorama_x_right = translations[i].x + projected_images[i].cols;
        }
        if (parorama_y_top < translations[i].y) {
            parorama_y_top = translations[i].y;
        }
        if (parorama_y_bottom > translations[i].y + projected_images[i].rows) {
            parorama_y_bottom = translations[i].y + projected_images[i].rows;
        }
    }
    int panorama_width = parorama_x_right - parorama_x_left;
    int panorama_height = parorama_y_bottom - parorama_y_top;

    // Create the final image
    auto panorama_image = cv::Mat(panorama_height, panorama_width, projected_images[0].type());
    for (int i = 0; i < projected_images.size(); i++) {
        auto source_region = cv::Rect(0, parorama_y_top - translations[i].y,
                                      projected_images[i].cols, panorama_height);
        auto destination_region = cv::Rect(translations[i].x + parorama_x_left, 0,
                                           projected_images[i].cols, panorama_height);
        projected_images[i](source_region).copyTo(panorama_image(destination_region));
    }

    return panorama_image;
}