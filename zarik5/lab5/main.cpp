#include "panoramic_image.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

struct SequenceInfo {
    std::string glob_path;
    std::string output_path;
    float half_fov_deg;
    float distance_ratio;
};

int main() {
    const float DISTANCE_RATIO = 3;

    std::filesystem::create_directory("../../../output");

    std::vector<SequenceInfo> sequences_infos = {
        {"../../../data/dataset_dolomites/dolomites/*.png", "../../../output/dolomites.png", 27,
         DISTANCE_RATIO},
        {"../../../data/dataset_kitchen/kitchen/*.bmp", "../../../output/kitchen.png", 33,
         DISTANCE_RATIO},
        {"../../../data/dataset_lab/data/*.bmp", "../../../output/lab.png", 33, DISTANCE_RATIO},
        {"../../../data/dataset_lab_19_automatic/*.png", "../../../output/lab_19_automatic.png", 33,
         DISTANCE_RATIO},
        {"../../../data/dataset_lab_19_manual/*.png", "../../../output/lab_19_manual.png", 33,
         DISTANCE_RATIO},
    };

    for (auto &info : sequences_infos) {
        std::vector<std::string> image_paths;
        cv::glob(info.glob_path, image_paths);

        // Note: here I expect images to be sorted from left to right
        std::vector<cv::Mat> images;
        for (auto &path : image_paths) {
            images.push_back(cv::imread(path));
        }

        auto panorama_image = PanoramicImage(images).stitch(info.half_fov_deg, info.distance_ratio);

        cv::imshow(info.glob_path, panorama_image);
        cv::waitKey();

        cv::imwrite(info.output_path, panorama_image);
    }
}
