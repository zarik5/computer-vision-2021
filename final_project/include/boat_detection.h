
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

class BoatDetector
{

public:
  BoatDetector(cv::String HOG_xml_file);
  BoatDetector();
  void train(cv::String images_folder,
             cv::String ground_truth_folder);

  void save(cv::String HOG_xml_file);
  void load(cv::String HOG_xml_file);

  std::vector<cv::Rect> detectBoats(cv::Mat input_img);

  std::vector<cv::Rect> NMS(std::vector<cv::Rect> ships,
                            std::vector<double> probabilities);

private:
  void loadImages(cv::String images_folder,
                  cv::String ground_truth_folder,
                  std::vector<cv::Mat> &training_images,
                  std::vector<std::vector<cv::Rect>> &positive_labels);

  void randomNegatives(std::vector<cv::Mat> training_images,
                       std::vector<std::vector<cv::Rect>> positive_rects,
                       cv::Mat& training_HOGs);

  void mineNegatives(cv::Mat training_image,
                     std::vector<cv::Rect> gound_truths,
                     std::vector<cv::Rect> found_boats,
                     cv::Mat &training_HOGs);
  void SVM_to_HOG_converter(cv::Ptr<cv::ml::SVM> SVM);

  float threshold_IoU = 0.1;
  cv::Ptr<cv::HOGDescriptor> HOG_descriptor;
};