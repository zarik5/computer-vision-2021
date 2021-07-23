#include <boat_detection.h>

const std::string TRAINING_HELP_MESSAGE = R"help(
boat_train: Start the training in python 

USAGE:
final_project boat_train <images_dir> <labels_dir> <model_dir> <epoch_max>
Note: the dataset selected id the COCCO with the YOLO labels format.
)help";

const std::string DETECTION_HELP_MESSAGE = R"help(
boat_detection: Detect boats 

USAGE:
final_project boat_detect <model> <images_dir> <ground_truth> <output_dir>
)help";

boat_detection::boat_detection() {}

/**
 * Function used to train the net in with the python script
 *
 * @param arguments Parameters given in input as <images_dir> <labels_dir> <model_dir> <epoch_max>
 */
void boat_detection::train(std::vector<std::string> arguments) {
    if (arguments.size() != 2) {
        std::cout << TRAINING_HELP_MESSAGE;
        return;
    }
    auto command = std::ostringstream();
    command << "python ../boat_train.py " << arguments[0] << " " << arguments[1] << " "
            << arguments[2] << " " << arguments[3];
    system(command.str().c_str());
}

/**
 * Function used to detect boats and confidences
 *
 * @param arguments Parameters given in input as <model> <images_dir> <ground_truth> <output_dir>
 */
void boat_detection::detect(std::vector<std::string> arguments) {
    if (arguments.size() != 2) {
        std::cout << DETECTION_HELP_MESSAGE;
        return;
    }

    load(cv::String(arguments[0]));
    std::vector<cv::Mat> images_vector;
    std::vector<std::vector<cv::Rect>> ground_truths;
    std::vector<cv::String> images_names;
    loadGroundTruth(arguments[1], arguments[2], images_names, images_vector, ground_truths);
    for (int i = 0; i < images_vector.size(); i++) {
        std::vector<float> confidences;
        std::vector<cv::Rect> boats_found;
        detectBoats(images_vector[i], boats_found, confidences);

        for (int j = 0; j < boats_found.size(); j++) {
            cv::rectangle(images_vector[i], boats_found[j], cv::Scalar(0, 0, 255), 3);
        }
        for (int j = 0; j < ground_truths[i].size(); j++) {
            cv::rectangle(images_vector[i], ground_truths[i][j], cv::Scalar(0, 255, 0), 3);
        }
        std::vector<float> scores_ground_truth;
        std::vector<float> scores_boats;
        scores(ground_truths[i], boats_found, scores_ground_truth, scores_boats);
        std::cout << images_names[i] << " done\n";

        writeIoU(arguments[3] + images_names[i] + ".txt", scores_ground_truth, scores_boats,
                 boats_found);
        cv::imwrite(arguments[3] + images_names[i] + ".jpg", images_vector[i]);
    }
}

/**
 * Function used to load the trained net from a file
 *
 * @param Net_pb_file File in .pb extention of the net trained
 */
void boat_detection::load(cv::String Net_pb_file) { net = cv::dnn::readNet(Net_pb_file); }


/**
 * Function used to detect boats and confidences
 *
 * @param input_img Image to analize
 * @param found_boats vector of boats found
 * @param confidences vector of confidences for each boat
 */
void boat_detection::detectBoats(cv::Mat input_img, std::vector<cv::Rect> &found_boats,
                                 std::vector<float> &confidences) {
    std::vector<cv::Rect> proposed_regions;
    proposalRegions(input_img, proposed_regions);
    std::vector<cv::Rect> found_boats_net;
    std::vector<float> confidences_net;

    for (cv::Rect boat : proposed_regions) {
        float ratio = float(MIN(boat.width, boat.width)) / float(MAX(boat.width, boat.width));
        if ((boat.width < threshold_width) | (boat.height < threshold_height) |
            (ratio < threshold_ratio)) {
            continue;
        }
        // cv::Scalar mean, std;
        // cv::Mat proposed_area;
        // input_img(boat).convertTo(proposed_area, CV_32FC3);
        // cv::meanStdDev(proposed_area,mean, std);
        // cv::subtract(proposed_area,mean,proposed_area);
        // cv::multiply(proposed_area, std, proposed_area);
        cv::Mat input_blob = cv::dnn::blobFromImage(input_img(boat), 1, cv::Size(224, 224));
        net.setInput(input_blob);
        cv::Mat out = net.forward();

        if (out.at<float>(0, 0) > threshold_confidence) {
            found_boats_net.push_back(boat);
            confidences_net.push_back(out.at<float>(0, 0));
        }
    }
    std::vector<cv::Rect> first_step_boats;
    std::vector<float> first_step_confidences;
    NMS(found_boats_net, confidences_net, first_step_boats, first_step_confidences);

    shiftSuppresion(input_img, first_step_boats, first_step_confidences, found_boats, confidences);
}

/**
 * Function to compute the scores between ground thruth and boats found
 *
 * @param gound_truths vector of ground truth
 * @param found_boats vector of boats found
 * @param scores_ground_truth vector of best IoU for each ground truth
 * @param scores_boats vector of best IoU for each boat
 */
void boat_detection::scores(std::vector<cv::Rect> gound_truths, std::vector<cv::Rect> found_boats,
                            std::vector<float> &scores_ground_truth,
                            std::vector<float> &scores_boats) {
    scores_ground_truth = std::vector<float>(gound_truths.size(), 0);
    scores_boats = std::vector<float>(found_boats.size(), 0);
    for (size_t i = 0; i < found_boats.size(); i++) {
        for (size_t j = 0; j < gound_truths.size(); j++) {
            float IoU = IoUCompute(gound_truths[j], found_boats[i]);

            if (IoU > scores_ground_truth[j]) {
                scores_ground_truth[j] = IoU;
            }
            if (IoU > scores_boats[i]) {
                scores_boats[i] = IoU;
            }
        }
    }
}

/**
 * Non Maxima Suppresion
 *
 * @param found_boats_net boats found by the net
 * @param confidences_net confidences found by the net
 * @param boats returned boats from the highest probability
 * @param confidences vector of confidences for each boat returned
 */
void boat_detection::NMS(std::vector<cv::Rect> found_boats_net, std::vector<float> confidences_net,
                         std::vector<cv::Rect> &boats, std::vector<float> &confidences) {
    if (!found_boats_net.empty()) {
        std::vector<int> indeces;
        std::vector<cv::Rect> shipsGood;
        cv::sortIdx(confidences_net, indeces, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
        for (size_t i = 0; i < found_boats_net.size(); i++) {
            int index_i = indeces[i];
            if (confidences_net[index_i] == 0) {
                continue;
            }
            for (size_t j = i + 1; j < found_boats_net.size(); j++) {
                int index_j = indeces[j];
                if (confidences_net[index_i] > confidences_net[index_j]) {
                    float IoU = IoUCompute(found_boats_net[index_i], found_boats_net[index_j]);
                    if (IoU > threshold_IoU) {
                        confidences_net[index_j] = 0;
                    }
                }
            }
            boats.push_back(found_boats_net[index_i]);
            confidences.push_back(confidences_net[index_i]);
        }
    }
}


/**
 * Check if the shifting the region there is still continuity in the prediction, removing possible false positives isolated
 *
 * @param gound_truths Image to analize
 * @param found_boats vector of boats found
 * @param scores_ground_truth vector of confidences for each boat
 * @param scores_boats vector of confidences for each boat
 */
void boat_detection::shiftSuppresion(cv::Mat input_img,std::vector<cv::Rect> found_boats_net, std::vector<float> confidences_net, std::vector<cv::Rect> &boats, std::vector<float> &confidences)
{
    for (size_t i = 0; i < found_boats_net.size(); i++)
    {
        float count = 0;
        float max_count = 0;
        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                if((y==0)&&(x==0))
                {
                    continue;
                }
                max_count++;
                cv::Rect shifted_boat = found_boats_net[i] + cv::Point(x * found_boats_net[i].width / 3, y * found_boats_net[i].height / 3);
                if((shifted_boat & cv::Rect(0, 0, input_img.cols, input_img.rows)) != shifted_boat){
                    count++;
                    continue;
                }
                cv::Mat input_blob = cv::dnn::blobFromImage(input_img(shifted_boat),1,cv::Size(224, 224));
                net.setInput(input_blob);
                cv::Mat out = net.forward();

                if (out.at<float>(0, 0) > 0.5)
                {
                    count++;
                }
            }
        }
        if(count/max_count>0.3){
            boats.push_back(found_boats_net[i]);
            confidences.push_back(confidences_net[i]);                        
        }
    }
}




/**
 * Function to compute the IoU
 *
 * @param rect1 First rect
 * @param rect2 Second rect
 */
float boat_detection::IoUCompute(cv::Rect rect1, cv::Rect rect2) {
    float intersection_area = (rect1 & rect2).area();
    float union_area = rect1.area() + rect2.area() - intersection_area;
    float IoU = intersection_area / union_area;
    return IoU;
}

/**
 * Segmentation of the image to extract the proposed regions
 *
 * @param input_img Image to analize
 * @param proposedRegions vector of proposed regions
 */
void boat_detection::proposalRegions(cv::Mat input_img, std::vector<cv::Rect> &proposedRegions) {

    cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss =
        cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
    ss->setBaseImage(input_img);

    ss->switchToSelectiveSearchFast();
    ss->process(proposedRegions);
}

/**
 * Loader of images from the folders given in input
 *
 * @param test_images_folder folder of test images
 * @param ground_truth_folder folder of ground truths
 * @param images_names names of single files
 * @param images_vector vector of images loaded
 * @param ground_truths vector of ground truth associated
 */
void boat_detection::loadGroundTruth(cv::String test_images_folder, cv::String ground_truth_folder,
                                     std::vector<cv::String> &images_names,
                                     std::vector<cv::Mat> &images_vector,
                                     std::vector<std::vector<cv::Rect>> &ground_truths) {
    std::vector<cv::String> image_names_vector;
    cv::glob(test_images_folder + "*", image_names_vector);

    for (cv::String image_name : image_names_vector) {
        cv::Mat image = cv::imread(image_name);
        if (image.empty()) {
            continue;
        }

        images_vector.push_back(image);
        int slash_pos = image_name.find_last_of("\\/") + 1;
        int point_pos = image_name.find_last_of(".");
        images_names.push_back(image_name.substr(slash_pos, point_pos - slash_pos));
        cv::String fileName = ground_truth_folder + images_names.back() + ".txt";
        std::ifstream ground_truths_file = std::ifstream(fileName.c_str());

        std::vector<cv::Rect> image_ground_truths;
        for (std::string line; std::getline(ground_truths_file, line);) {
            std::vector<int> values(4, 0);
            std::size_t end = line.find(":");
            if (line.substr(0, end).compare("boat") != 0) {
                continue;
            }
            for (size_t i = 0; i < 4; i++) {
                std::size_t start = end + 1;
                end = line.find(";", start);
                values[i] = std::atoi(line.substr(start, end - start).c_str());
            }
            image_ground_truths.push_back(
                cv::Rect(cv::Point(values[0], values[2]), cv::Point(values[1], values[3])));
        }
        ground_truths.push_back(image_ground_truths);
    }
}

/**
 * Utility function used to write the results
 *
 * @param output_file file to write
 * @param scores_ground_truth best scores for each ground truth
 * @param scores_boats best scores for esch boat
 * @param boats_found boats founded in the image
 */
void boat_detection::writeIoU(cv::String output_file, std::vector<float> scores_ground_truth,
                              std::vector<float> scores_boats, std::vector<cv::Rect> boats_found) {
    std::ofstream scores_file(output_file);
    if (scores_file.is_open()) {
        scores_file << "Founded boats scores:\n";
        for (size_t i = 0; i < scores_boats.size(); i++) {
            scores_file << std::to_string(scores_boats[i]) << ": "
                        << std::to_string(boats_found[i].x) << " "
                        << std::to_string(boats_found[i].br().x) << " "
                        << std::to_string(boats_found[i].y) << " "
                        << std::to_string(boats_found[i].br().y) << "\n";
        }
        scores_file << "Ground thruth best scores:\n";
        for (size_t i = 0; i < scores_ground_truth.size(); i++) {
            scores_file << std::to_string(scores_ground_truth[i]) << "\n";
        }
        scores_file.close();
    }
}