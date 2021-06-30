#include "boat_detection.h"

#include <iostream>
#include <opencv2/opencv.hpp>

const std::string TRAIN_HELP_MESSAGE = R"help(
boat_train: Train a bag of words for boat and non-boat classes

USAGE:
final_project boat_train <in_images_dir> <in_annotations_file> <out_histograms_file>
)help";

const std::string DETECT_HELP_MESSAGE = R"help(
boat_detect: Find bounding-boxes around boats in an image

USAGE:
final_project boat_detect <in_histograms_file> <in_image_file> <out_bounding_boxes_file> [--display]

FLAG:
display: Show the detection result in a window before closing the program
)help";

void boat_detection::train(std::vector<std::string> arguments) {
    if (arguments.size() != 3) {
        std::cout << TRAIN_HELP_MESSAGE;
        return;
    }

    auto images_dir = arguments[0];
    auto annotations_file = arguments[1];
    auto histograms_file = arguments[2];
}

void boat_detection::detect(std::vector<std::string> arguments) {
    if (arguments.size() >= 3 && arguments.size() <= 4) {

        // todo
    } else {
        std::cout << DETECT_HELP_MESSAGE;
    }

    auto histograms_file = arguments[0];
    auto image_file = arguments[1];
    auto annotations_file = arguments[2];
    bool should_display = false;
    if (arguments.size() == 4) {
        if (arguments[3] == "--display") {
            should_display = true;
        } else {
            std::cout << DETECT_HELP_MESSAGE;
            return;
        }
    }
}