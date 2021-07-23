#include <iostream>
#include <vector>

#include "boat_detection.h"
#include "sea_segmentation.h"

const std::string HELP_MESSAGE = R"help(
final_project: detect boats or segment the sea in an image

USAGE:
final_project <step> [<step args>]

STEPS: boat_train | boat_detect | sea_train | sea_segment
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
            boat_detection boatFinder;
            boatFinder.train(arguments);
        } else if (step == "boat_detect") {
            boat_detection boatFinder;
            boatFinder.detect(arguments);
        } else if (step == "sea_train") {
            sea_segmentation::train(arguments);
        } else if (step == "sea_segment") {
            sea_segmentation::segment_image(arguments);
        } else {
            std::cout << HELP_MESSAGE;
        }
    } else {
        std::cout << HELP_MESSAGE;
    }

    std::cout << std::endl << "Done" << std::endl;
}