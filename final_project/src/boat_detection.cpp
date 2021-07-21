#include <boat_detection.h>
#include <time.h>

#define MAX_NEGATIVES 2
#define MAX_TRIES 20
#define MAX_HNM_ITERATIONS 3
#define MAX_TRHESHOLD_IOU 1

BoatDetector::BoatDetector(cv::String HOG_xml_file) { load(HOG_xml_file); }

BoatDetector::BoatDetector()
{
    HOG_descriptor = cv::Ptr<cv::HOGDescriptor>(
        new cv::HOGDescriptor(cv::Size(68, 68), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9,
                              1, -1., cv::HOGDescriptor::L2Hys, .2000000000000000111, true));
}

void BoatDetector::load(cv::String HOG_xml_file)
{
    HOG_descriptor = cv::Ptr<cv::HOGDescriptor>(new cv::HOGDescriptor(
        cv::Size(68, 68), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9,
        1, -1., cv::HOGDescriptor::L2Hys, .2000000000000000111, true));
    HOG_descriptor->load(HOG_xml_file);
}

void BoatDetector::save(cv::String HOG_xml_file) { HOG_descriptor->save(HOG_xml_file); }

void BoatDetector::train(cv::String images_folder, cv::String ground_truth_folder)
{

    std::vector<cv::Mat> training_images;
    std::vector<std::vector<cv::Rect>> positive_patchs;
    cv::Mat training_HOGs;
    loadImages(images_folder, ground_truth_folder, training_images, positive_patchs, training_HOGs);

    std::vector<int> training_labels(training_HOGs.rows, 1);

    randomNegatives(training_images, positive_patchs, training_HOGs);
    training_labels.insert(training_labels.end(), training_HOGs.rows - training_labels.size(), -1);

    cv::Ptr<cv::ml::SVM> linear_svm = cv::ml::SVM::create();
    linear_svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1e4, 1e-4));
    linear_svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
    linear_svm->setType(cv::ml::SVM::C_SVC);
    cv::Ptr<cv::ml::TrainData> training_data = cv::ml::TrainData::create(training_HOGs, cv::ml::ROW_SAMPLE, training_labels);

    training_data->shuffleTrainTest();
    linear_svm->trainAuto(training_data);
    cv::Mat responses;
    save(".\\HOG_Step_0.xml");
    std::cout << std::to_string(linear_svm->calcError(training_data, false, responses)) << "\n";

    for (size_t i = 0; i < MAX_HNM_ITERATIONS; i++)
    {

        SVM_to_HOG_converter(linear_svm);

        for (size_t j = 0; j < training_images.size(); j++)
        {

            std::vector<cv::Rect> ships_found = detectBoats(training_images[j]);
            mineNegatives(training_images[j], positive_patchs[j], ships_found, training_HOGs, MAX_TRHESHOLD_IOU * i / MAX_HNM_ITERATIONS);
            training_labels.insert(training_labels.end(), training_HOGs.rows - training_labels.size(), -1);

            std::cout << std::to_string(j) << "   ";
        }
        training_data = cv::ml::TrainData::create(training_HOGs, cv::ml::ROW_SAMPLE, training_labels);

        training_data->shuffleTrainTest();
        linear_svm->train(training_data);
        cv::Mat responses;
        std::cout << std::to_string(linear_svm->calcError(training_data, false, responses)) << "\n";
        save(".\\HOG_Step_" + std::to_string(i + 1) + ".xml");
    }
}

void BoatDetector::loadImages(cv::String images_folder, cv::String ground_truth_folder,
                              std::vector<cv::Mat> &training_images,
                              std::vector<std::vector<cv::Rect>> &positive_labels, cv::Mat &training_HOGs)
{

    std::vector<cv::String> image_names_vector;
    cv::glob(images_folder + "\\*", image_names_vector);

    for (int i = 0; i < image_names_vector.size(); i++)
    {
        cv::String image_name = image_names_vector[i];
        cv::Mat image = cv::imread(image_name);
        if (image.empty())
        {
            continue;
        }
        cv::Mat padded_image;
        cv::copyMakeBorder(image, padded_image, image.cols / 2,
                           image.cols / 2, image.rows / 2,
                           image.rows / 2, cv::BORDER_CONSTANT);

        training_images.push_back(padded_image);
        int slash_pos = image_name.find_last_of("\\/");
        int point_pos = image_name.find_last_of(".");
        cv::String fileName =
            ground_truth_folder + image_name.substr(slash_pos, point_pos - slash_pos) + ".txt";
        std::ifstream train_image_file = std::ifstream(fileName.c_str());

        std::vector<cv::Rect> image_labels;
        for (std::string line; std::getline(train_image_file, line);)
        {
            std::vector<int> values(4, 0);
            std::size_t end = line.find(":");
            if (line.substr(0, end).compare("boat") != 0)
            {
                continue;
            }
            for (size_t i = 0; i < 4; i++)
            {
                std::size_t start = end + 1;
                end = line.find(";", start);
                values[i] = std::atoi(line.substr(start, end - start).c_str());
            }
            int width = values[1] - values[0];
            int height = values[3] - values[2];
            int max = MAX(width, height);
            cv::Rect padded_label(image.rows / 2 + values[0] + (width - max) / 2,
                                  image.cols / 2 + values[2] + (height - max) / 2,
                                  max, max);

            image_labels.push_back(padded_label);
            std::vector<float> descriptors;
            cv::Mat resized_image;
            cv::resize(padded_image(padded_label), resized_image, cv::Size(96, 96));
            HOG_descriptor->compute(resized_image, descriptors, cv::Size(8, 8), cv::Size(0, 0));
            training_HOGs.push_back(cv::Mat(descriptors).t());
        }
        positive_labels.push_back(image_labels);
    }
}

void BoatDetector::randomNegatives(std::vector<cv::Mat> training_images, std::vector<std::vector<cv::Rect>> positive_rects, cv::Mat &training_HOGs)
{

    cv::RNG rng(time(0));

    for (size_t i = 0; i < training_images.size(); i++)
    {
        int negatives_count = 0;
        int tries = 0;
        std::vector<cv::Rect> negative_image_labels;

        while ((negatives_count < MAX_NEGATIVES) & (tries < MAX_TRIES))
        {
            int x = rng.uniform(0, training_images[i].cols - 96);
            int y = rng.uniform(0, training_images[i].rows - 96);
            int min = MIN(training_images[i].cols - x, training_images[i].rows - y);

            int w = rng.uniform(96, min);
            cv::Rect possible_negative(x, y, w, w);

            for (cv::Rect positive : positive_rects[i])
            {
                float overlapping_area = (positive & possible_negative).area();

                if (overlapping_area > 0)
                {
                    tries++;
                    continue;
                }
            }
            std::vector<float> descriptors;
            cv::Mat resized_image;
            cv::resize(training_images[i](possible_negative), resized_image, cv::Size(96, 96));
            HOG_descriptor->compute(resized_image, descriptors, cv::Size(8, 8), cv::Size(0, 0));
            training_HOGs.push_back(cv::Mat(descriptors).t());
            negatives_count++;
            tries = 0;
        }
    }
}

std::vector<cv::Rect> BoatDetector::detectBoats(cv::Mat input_img)
{

    std::vector<cv::Rect> rects_boats;
    std::vector<double> distances;

    HOG_descriptor->detectMultiScale(input_img, rects_boats, distances);
    return NMS(rects_boats, distances);
}

std::vector<cv::Rect> BoatDetector::NMS(std::vector<cv::Rect> ships, std::vector<double> distances)
{

    std::vector<cv::Rect> shipsGood;
    if (!ships.empty())
    {
        std::vector<int> indeces;
        cv::sortIdx(distances, indeces, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
        for (size_t i = 0; i < ships.size(); i++)
        {
            int index_i = indeces[i];
            if (distances[index_i] == 0)
            {
                continue;
            }
            for (size_t j = i + 1; j < ships.size(); j++)
            {
                int index_j = indeces[j];
                int intersection_area = (ships[index_i] & ships[index_j]).area();
                int union_area = ships[index_i].area() + ships[index_j].area() - intersection_area;
                if (intersection_area / union_area > threshold_IoU)
                {
                    distances[index_j] = 0;
                }
            }
            shipsGood.push_back(ships[index_i]);
        }
    }
    return shipsGood;
}

void BoatDetector::mineNegatives(cv::Mat training_image, std::vector<cv::Rect> gound_truths, std::vector<cv::Rect> found_boats, cv::Mat &training_HOGs, float threshold_IoU_step)
{
    for (cv::Rect boat : found_boats)
    {
        for (cv::Rect thruth : gound_truths)
        {
            float intersection_area = (thruth & boat).area();
            float union_area = thruth.area() + boat.area() - intersection_area;
            if ((intersection_area / union_area) < threshold_IoU_step)
            {
                std::vector<float> descriptors;
                cv::Mat resized_image;
                cv::resize(training_image(boat), resized_image, cv::Size(96, 96));
                HOG_descriptor->compute(resized_image, descriptors, cv::Size(8, 8), cv::Size(0, 0));
                training_HOGs.push_back(cv::Mat(descriptors).t());
                continue;
            }
        }
    }
}

void BoatDetector::SVM_to_HOG_converter(cv::Ptr<cv::ml::SVM> SVM)
{
    cv::Mat support_vectors = SVM->getSupportVectors();

    cv::Mat alpha, svidx;
    float rho = (float)SVM->getDecisionFunction(0, alpha, svidx);

    std::vector<float> SVM_detector(support_vectors.cols + 1);
    memcpy(&SVM_detector[0], support_vectors.ptr(), support_vectors.cols * sizeof(SVM_detector[0]));
    SVM_detector[support_vectors.cols] = -rho;
    HOG_descriptor->setSVMDetector(SVM_detector);
}