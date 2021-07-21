#include <iostream>
#include <vector>

#include <boat_detection.h>

int main(int argc, char *argv[])
{
    

    if (std::strcmp(argv[1],"Training")==0)
    {
        BoatDetector boatFinder;
        boatFinder.train(cv::String(argv[2]),cv::String( argv[3]));
        boatFinder.save(".\\HOG_Params.xml");
    }
    else
    {
        BoatDetector boatFinder;
        boatFinder.load(cv::String(argv[2]));
        std::vector<cv::String> image_names;
        cv::String ia = cv::String(argv[3]) + "\\*";
        cv::glob(ia, image_names);

        for (cv::String image_name : image_names)
        {

            cv::Mat image = cv::imread(image_name,cv::IMREAD_GRAYSCALE);
            std::vector<cv::Rect> ships_found = boatFinder.detectBoats(image);

            cv::RNG rng(time(0));
            for (cv::Rect rect : ships_found)
            {
                cv::rectangle(image, rect, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 3);
            }

            cv::imshow("Boats found: " + std::to_string(ships_found.size()), image);
            cv::waitKey();
        }
    }

    std::cout << std::endl
              << "Done" << std::endl;
}