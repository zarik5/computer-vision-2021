
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
#include <opencv2/dnn.hpp>

class boat_detection
{

public:
  boat_detection();
  void train(std::vector<std::string> arguments);
  void detect(std::vector<std::string> arguments);

  void load(cv::String Net_pb_file);

  void detectBoats(cv::Mat input_img, std::vector<cv::Rect> &found_boats, std::vector<float> &confidences);
  void scores(std::vector<cv::Rect> gound_truths, std::vector<cv::Rect> found_boats, std::vector<float> &scores_ground_truth, std::vector<float> &scores_boats);
  void loadGroundTruth(cv::String test_images_folder, cv::String ground_truth_folder, std::vector<cv::String> &images_names, std::vector<cv::Mat> &images_vector, std::vector<std::vector<cv::Rect>> &ground_truths);
  void writeIoU(cv::String image_name, std::vector<float> scores_ground_truth, std::vector<float> scores_boats, std::vector<cv::Rect> boats_found);

private:
  float IoUCompute(cv::Rect rect1, cv::Rect rect2);
  void proposalRegions(cv::Mat input_img, std::vector<cv::Rect> &proposedRegions);
  void NMS(std::vector<cv::Rect> found_boats_net, std::vector<float> confidences_net, std::vector<cv::Rect> &found_boats, std::vector<float> &confidences);
  void shiftSuppresion(cv::Mat input_img,std::vector<cv::Rect> found_boats_net, std::vector<float> confidences_net, std::vector<cv::Rect> &boats, std::vector<float> &confidences);
  float threshold_IoU = 0.1;
  float threshold_confidence = 0.8;
  int threshold_width = 15;
  int threshold_height = 15;
  float threshold_ratio = 0.1;
  cv::dnn::Net net;
};
