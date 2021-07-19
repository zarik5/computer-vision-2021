#include <iostream>
#include <vector>

#include "boat_detection.h"
#include "sea_segmentation.h"

const std::string HELP_MESSAGE = R"help(
final_project: detect boats or segment the sea in an image

USAGE:
final_project <step> [<step args>]

STEPS: boat_train | boat_detect | sea_prep_data | sea_segment
You can read help messages for each step by typing `<executable> <step>`.

)help";

int main(int argc, char *argv[]) {
    if (argc >= 2) {
        std::string step = argv[1];
        std::vector<std::string> arguments;
        for (int i = 2; i < argc; i++) {
            arguments.push_back(argv[i]);
        }

        if (step == "boat_train") {
            boat_detection::train(arguments);
        } else if (step == "boat_detect") {
            boat_detection::detect(arguments);
        } else if (step == "sea_prep_data") {
            sea_segmentation::prepare_dataset(arguments);
        } else if (step == "sea_prep_image") {
            sea_segmentation::prepare_image(arguments);
        } else if (step == "sea_show_segm") {
            sea_segmentation::show_segmentation(arguments);
        } else {
            std::cout << HELP_MESSAGE;
        }
    } else {
        std::cout << HELP_MESSAGE;
    }

    std::cout << std::endl << "Done" << std::endl;
}