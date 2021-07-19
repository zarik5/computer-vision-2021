#include <iostream>
#include <vector>

#include <boat_detection.h>

int main(int argc, char *argv[]) {
    BoatDetector boatFinder;
    if (argv[1] == "Training") {
        boatFinder.train(argv[2], argv[3]);

        boatFinder.save(".\\HoggSVR.xml");
    } else {
        std::vector<cv::String> image_names;
        cv::glob(argv[3], image_names);

        for (cv::String image_name : image_names) {

            cv::Mat image = cv::imread(image_name);
            std::vector<cv::Rect> ships_found = boatFinder.detectBoats(image);

            cv::RNG rng(time(0));
            for (cv::Rect rect : ships_found) {
                cv::rectangle(image, rect, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 3);
            }
            cv::imshow("Boats found: " + std::to_string(ships_found.size()), image);

            cv::waitKey();
        }
    }

    std::cout << std::endl << "Done" << std::endl;
}