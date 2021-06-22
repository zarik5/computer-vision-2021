#include <iostream>
#include <vector>

#include "boat_detection.h"
#include "sea_segmentation.h"

const std::string HELP_MESSAGE = R"help(
USAGE:
<executable> <step> [<step args>]

STEPS:  boat_train | boat_detect | sea_preproc | sea_segment
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
        } else if (step == "sea_preproc") {
            sea_segmentation::prepare_dataset(arguments);
        } else if (step == "sea_segment") {
            sea_segmentation::segment(arguments);
        } else {
            std::cout << HELP_MESSAGE;
        }
    } else {
        std::cout << HELP_MESSAGE;
    }
}