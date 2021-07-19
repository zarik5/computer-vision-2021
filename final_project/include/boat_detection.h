
// Giacomello Mattia
// I.D. 1210988

#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>

class BoatDetector {

  public:
    BoatDetector(cv::String MLP_xml_file, cv::String BoW_mat_file);
    BoatDetector();
    void train(cv::String images_folder, cv::String ground_truth_folder, int histogram_clusters);

    void save(cv::String BoW_mat_file, cv::String MLP_xml_file);
    void load(cv::String BoW_mat_file, cv::String MLP_xml_file);

    std::vector<cv::Rect> detectBoats(cv::Mat input_img, int proposed_regions);
    std::vector<cv::Rect> segmentationKDRP(cv::Mat img, int proposed_regions);

    std::vector<cv::Rect> NMS(std::vector<cv::Rect> ships, std::vector<float> probabilities);

    float testResult(std::vector<cv::Rect> found_boats, std::vector<cv::Rect> gound_truths);

  private:
    void loadImages(cv::String images_folder, cv::String ground_truth_folder,
                    std::vector<cv::Mat> &training_images,
                    std::vector<std::vector<cv::Rect>> &positive_labels);

    void negativeMining(std::vector<cv::Mat> training_images,
                        std::vector<std::vector<cv::Rect>> positive_labels,
                        std::vector<std::vector<cv::Rect>> &negative_labels);

    int histogram_clusters;
    float threshold_IoU = 0.1;
    cv::Ptr<cv::ml::SVM> svm;
    cv::Ptr<cv::ml::ANN_MLP> mlp;
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::BOWImgDescriptorExtractor> bow;
};