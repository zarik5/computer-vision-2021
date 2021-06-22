#include "sea_segmentation.h"

#include <iostream>
#include <opencv2/opencv.hpp>

const std::string PREPROC_HELP_MESSAGE = R"help(
sea_preproc: Extract square windows of pixels and their class (sea or non-sea)

USAGE:
<executable> sea_preproc <in_images_dir> <in_annotations_dir> <out_images_dir> <out_classes_file>
)help";

const std::string SEGMENT_HELP_MESSAGE = R"help(
sea_segment: Segment the sea from an image

USAGE:
<executable> sea_segment <in_model_file> <in_image_file> <out_annotation_file> [--display]

FLAG:
display: Show the segmentation result in a window before closing the program
)help";

namespace sea_segmentation {
void prepare_dataset(std::vector<std::string> arguments) {
    if (arguments.size() == 4) {
        auto in_images_dir = arguments[0];
        auto annotations_dir = arguments[1];
        auto out_images_dir = arguments[2];
        auto classes_file = arguments[3];

        // todo
    } else {
        std::cout << PREPROC_HELP_MESSAGE;
    }
}

void segment(std::vector<std::string> arguments) {
    if (arguments.size() >= 3 && arguments.size() <= 4) {
        auto model_file = arguments[0];
        auto image_file = arguments[1];
        auto annotation_file = arguments[2];
        bool should_display = false;
        if (arguments.size() == 4) {
            if (arguments[3] == "--display") {
                should_display = true;
            } else {
                std::cout << SEGMENT_HELP_MESSAGE;
                return;
            }
        }

        // todo
    } else {
        std::cout << SEGMENT_HELP_MESSAGE;
    }
}

} // namespace sea_segmentation