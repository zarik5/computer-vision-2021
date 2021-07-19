#include <boat_detection.h>
#include <time.h>

BoatDetector::BoatDetector(cv::String HOG_xml_file)
{
    load(HOG_xml_file);
}

BoatDetector::BoatDetector()
{
    HOG_descriptor = cv::Ptr<cv::HOGDescriptor>(new cv::HOGDescriptor(cv::Size(96, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9, 1, -1., cv::HOGDescriptor::L2Hys, .2000000000000000111, true));
}

void BoatDetector::load(cv::String HOG_xml_file)
{
    HOG_descriptor = cv::Ptr<cv::HOGDescriptor>(new cv::HOGDescriptor(cv::Size(96, 96), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9));
    HOG_descriptor->load(HOG_xml_file);
}

void BoatDetector::save(cv::String HOG_xml_file)
{
    HOG_descriptor->save(HOG_xml_file);
}

void BoatDetector::train(cv::String images_folder,
                         cv::String ground_truth_folder)
{

    std::vector<cv::Mat> training_images;
    std::vector<std::vector<cv::Rect>> positive_labels;
    loadImages(images_folder, ground_truth_folder, training_images, positive_labels);

    cv::Mat training_HOGs;

    for (size_t i = 0; i < training_images.size(); i++)
    {
        cv::Mat padded_image;
        cv::copyMakeBorder(training_images[i], padded_image, training_images[i].cols / 2, training_images[i].cols / 2, training_images[i].rows / 2, training_images[i].rows / 2, cv::BORDER_CONSTANT);
        for (cv::Rect label : positive_labels[i])
        {

            int max = MAX(label.width, label.height);
            cv::Rect padded_label(training_images[i].rows / 2 + label.x + (label.width - max) / 2, training_images[i].cols / 2 + label.y + (label.height - max) / 2, max, max);
            std::vector<float> descriptors;
            cv::Mat resized_image;
            cv::resize(padded_image(padded_label), resized_image, cv::Size(96, 96));
            HOG_descriptor->compute(resized_image, descriptors, cv::Size(8, 8), cv::Size(0, 0));
            training_HOGs.push_back(cv::Mat(descriptors).t());
        }
        cv::waitKey();
    }

    std::vector<int> training_labels;

    training_labels.insert(training_labels.end(), training_HOGs.rows, 1);
    randomNegatives(training_images, positive_labels, training_HOGs);
    training_labels.insert(training_labels.end(), training_HOGs.rows - training_labels.size(), -1);

    int max_HNM = 3;
    int cannato = 1;

    cv::Ptr<cv::ml::SVM> linear_svm = cv::ml::SVM::create();

    linear_svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1e5, 1e-5));
    linear_svm->setKernel(cv::ml::SVM::LINEAR);
    linear_svm->setType(cv::ml::SVM::EPS_SVR);
    linear_svm->setP(0.1);
    linear_svm->setC(0.01);

    for (size_t i = 0; (i < max_HNM) & (cannato != 0); i++)
    {

        linear_svm->train(training_HOGs, cv::ml::ROW_SAMPLE, training_labels);
        //0,103
        std::cout << "\n"
                  << std::to_string(linear_svm->getKernelType()) << "   " << std::to_string(linear_svm->getType()) << "\n";

        SVM_to_HOG_converter(linear_svm);
        cannato = 0;
        for (size_t j = 0; j < training_images.size(); j++)
        {
            std::vector<cv::Rect> ships_found = detectBoats(training_images[j]);
            mineNegatives(training_images[j], positive_labels[j], ships_found, training_HOGs);
            int mispoken = training_HOGs.rows - training_labels.size();
            cannato += mispoken;

            training_labels.insert(training_labels.end(), mispoken, -1);
            if (int(j * 100 / training_images.size()) % 5 == 0) {
                std::cout << std::to_string(j * 100 / training_images.size()) << "%  ";
            }
        }
        std::cout << "\n" << std::to_string(i) << "  " << std::to_string(cannato) << "  " << std::to_string(training_HOGs.rows) << "\n";

        save(".\\hoggetSVC" + std::to_string(i) + ".xml");
    }
}

void BoatDetector::loadImages(cv::String images_folder,
                              cv::String ground_truth_folder,
                              std::vector<cv::Mat> &training_images,
                              std::vector<std::vector<cv::Rect>> &positive_labels)
{

    std::vector<cv::String> image_names_vector;
    cv::glob(images_folder + "*", image_names_vector);

    for (int i = 0; i < image_names_vector.size(); i++)
    {
        cv::String image_name = image_names_vector[i];
        cv::Mat image = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            continue;
        }
        cv::normalize(image, image);
        training_images.push_back(image);
        int slash_pos = image_name.find_last_of("\\/") + 1;
        int point_pos = image_name.find_last_of(".");
        cv::String fileName = ground_truth_folder + image_name.substr(slash_pos, point_pos - slash_pos) + ".txt";
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

            image_labels.push_back(cv::Rect(cv::Point(values[0], values[2]), cv::Point(values[1], values[3])));
        }
        positive_labels.push_back(image_labels);
    }
}

void BoatDetector::randomNegatives(std::vector<cv::Mat> training_images,
                                   std::vector<std::vector<cv::Rect>> positive_rects,
                                   cv::Mat &training_HOGs)
{
    cv::RNG rng;

    for (size_t i = 0; i < training_images.size(); i++)
    {
        int j = 0;
        int tries = 0;
        std::vector<cv::Rect> negative_image_labels;

        while (j < 4 & tries < 10)
        {
            int x = rng.uniform(0, training_images[i].cols - 96);
            int y = rng.uniform(0, training_images[i].rows - 96);
            int min = MIN(training_images[i].cols - x, training_images[i].rows - y);

            int w = rng.uniform(96, min);
            cv::Rect possible_negative(x, y, w, w);

            for (cv::Rect positive : positive_rects[i])
            {
                float overlapping_area = (positive & possible_negative).area();
                float union_area = positive.area() + possible_negative.area() - overlapping_area;

                if ((overlapping_area / union_area) > threshold_IoU)
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
            j++;
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

std::vector<cv::Rect> BoatDetector::NMS(std::vector<cv::Rect> ships,
                                        std::vector<double> distances)
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

void BoatDetector::mineNegatives(cv::Mat training_image,
                                 std::vector<cv::Rect> gound_truths,
                                 std::vector<cv::Rect> found_boats,
                                 cv::Mat &training_HOGs)
{
    for (cv::Rect boat : found_boats)
    {
        for (cv::Rect thruth : gound_truths)
        {
            float intersection_area = (thruth & boat).area();
            float union_area = thruth.area() + boat.area() - intersection_area;
            if ((intersection_area / union_area) < threshold_IoU)
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
    double rho = SVM->getDecisionFunction(0, alpha, svidx);

    std::vector<float> SVM_detector(support_vectors.cols + 1);
    memcpy(&SVM_detector[0], support_vectors.ptr(), support_vectors.cols * sizeof(SVM_detector[0]));
    SVM_detector[support_vectors.cols] = (float)-rho;
    HOG_descriptor->setSVMDetector(SVM_detector);
}