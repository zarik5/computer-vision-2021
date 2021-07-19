#include <iostream>
#include <vector>

#include <boat_detection.h>

int main(int argc, char *argv[]) {

    //    boat_detection::train(arguments[2],arguments[3]);
    std::vector<cv::String> image_names;

    cv::glob("D:\\Desktop\\Project Jack-Rick - Copia (4) - "
             "Copia\\data\\FINAL_DATASET\\TEST_DATASET\\kaggle\\*",
             image_names);
    BoatDetector boatFinder("D:\\Desktop\\Project Jack-Rick\\mlp.xml",
                            "D:\\Desktop\\Project Jack-Rick\\vocabulary.tiff");

    for (int i = image_names.size(); i < image_names.size(); i++) {
        cv::String image_name = image_names[i];
        cv::Mat image = cv::imread(image_name);
        std::vector<cv::Rect> shipsFoundMlp = boatFinder.detectBoats(image, 100);
        for (cv::Rect rect : shipsFoundMlp) {
            cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 3);
        }

        cv::imshow("Segmentation KDRP", image);
        cv::waitKey();
    }
    std::cout << std::endl << "Done" << std::endl;
}