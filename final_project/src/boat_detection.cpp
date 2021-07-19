#include <boat_detection.h>

BoatDetector::BoatDetector(cv::String MLP_xml_file, cv::String BoW_mat_file) {

    detector = cv::xfeatures2d::SurfFeatureDetector::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SurfDescriptorExtractor::create();
    bow = cv::Ptr<cv::BOWImgDescriptorExtractor>(
        new cv::BOWImgDescriptorExtractor(extractor, matcher));
    mlp = cv::ml::ANN_MLP::create();
    load(MLP_xml_file, BoW_mat_file);
}

BoatDetector::BoatDetector() {
    detector = cv::xfeatures2d::SurfFeatureDetector::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SurfDescriptorExtractor::create();

    bow = cv::Ptr<cv::BOWImgDescriptorExtractor>(
        new cv::BOWImgDescriptorExtractor(extractor, matcher));
    mlp = cv::ml::ANN_MLP::create();
}

void BoatDetector::load(cv::String MLP_xml_file, cv::String BoW_mat_file) {
    cv::Mat vocabulary = cv::imread(BoW_mat_file, cv::IMREAD_UNCHANGED);
    bow->setVocabulary(vocabulary);
    mlp = cv::Algorithm::load<cv::ml::ANN_MLP>(MLP_xml_file);
}

void BoatDetector::save(cv::String MLP_xml_file, cv::String BoW_mat_file) {
    cv::Mat vocabulary = bow->getVocabulary();
    cv::imwrite(BoW_mat_file, vocabulary);
    bow->setVocabulary(vocabulary);
    mlp->save(MLP_xml_file);
}

void BoatDetector::train(cv::String images_folder, cv::String ground_truth_folder,
                         int histogram_clusters) {
    std::vector<cv::Mat> training_images;
    std::vector<std::vector<cv::Rect>> positive_labels;
    loadImages(images_folder, ground_truth_folder, training_images, positive_labels);

    std::vector<std::vector<cv::Rect>> negative_labels;
    negativeMining(training_images,positive_labels, negative_labels);

    cv::BOWKMeansTrainer trainer(histogram_clusters);
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SurfDescriptorExtractor::create();

    for (size_t i = 0; i < training_images.size(); i++) {
        for (cv::Rect label : positive_labels[i]) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            detector->detect(training_images[i](label), keypoints);
            extractor->compute(training_images[i](label), keypoints, descriptors);
            if (descriptors.empty()) {
                continue;
            }
            trainer.add(descriptors);
        }

        for (cv::Rect label : negative_labels[i]) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            detector->detect(training_images[i](label), keypoints);
            extractor->compute(training_images[i](label), keypoints, descriptors);

            if (descriptors.empty()) {
                continue;
            }
            trainer.add(descriptors);
        }
    }

    cv::Mat vocabulary = trainer.cluster();
    bow->setVocabulary(vocabulary);

    cv::Mat input_data, input_data_labels;
    for (size_t i = 0; i < training_images.size(); i++) {
        for (cv::Rect label : positive_labels[i]) {
            cv::Mat histogram;
            std::vector<cv::KeyPoint> keypoints;
            detector->detect(training_images[i](label), keypoints);
            bow->compute(training_images[i](label), keypoints, histogram);
            if (histogram.rows == 0) {
                continue;
            }
            input_data.push_back(histogram);
            float data[2] = {1, -1};
            input_data_labels.push_back(cv::Mat(1, 2, CV_32FC1, data));
        }

        for (cv::Rect label : negative_labels[i]) {
            cv::Mat histogram;
            std::vector<cv::KeyPoint> keypoints;
            detector->detect(training_images[i](label), keypoints);
            bow->compute(training_images[i](label), keypoints, histogram);
            if (histogram.rows == 0) {
                continue;
            }
            input_data.push_back(histogram);
            float data[2] = {-1, 1};
            input_data_labels.push_back(cv::Mat(1, 2, CV_32FC1, data));
        }
    }

    int hidden_size = histogram_clusters * 2 / 3 + 1;
    cv::Mat layersSize = (cv::Mat_<int>(3, 1) << histogram_clusters, hidden_size, 2);
    mlp->setLayerSizes(layersSize);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 2, 1);
    mlp->setTrainMethod(cv::ml::ANN_MLP::RPROP);
    mlp->setTermCriteria(
        cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 1e5, 1e-5));
    mlp->train(input_data, cv::ml::ROW_SAMPLE, input_data_labels);
}

void BoatDetector::loadImages(cv::String images_folder, cv::String ground_truth_folder,
                              std::vector<cv::Mat> &training_images,
                              std::vector<std::vector<cv::Rect>> &positive_labels) {

    std::vector<cv::String> image_names;
    cv::glob(images_folder + "*", image_names);

    for (int i = 0; i < image_names.size(); i++) {
        cv::String image_name = image_names[i];
        cv::Mat image = cv::imread(image_name);
        if (image.empty()) {
            continue;
        }
        training_images.push_back(image);
        int slash = image_name.find_last_of("\\/") + 1;
        int point = image_name.find_last_of(".");
        cv::String fileName =
            ground_truth_folder + image_name.substr(slash, point - slash) + ".txt";
        std::ifstream train_image_rects = std::ifstream(fileName.c_str());
        std::vector<cv::Rect> image_labels;
        for (std::string line; std::getline(train_image_rects, line);) {
            std::vector<int> values(4, 0);
            std::size_t start = line.find(":") + 1;
            for (size_t i = 0; i < 4; i++) {
                std::size_t end = line.find(";", start);
                values[i] = std::atoi(line.substr(start, end - start).c_str());
                start = end + 1;
            }

            image_labels.push_back(
                cv::Rect(cv::Point(values[0], values[2]), cv::Point(values[1], values[3])));
        }
        positive_labels.push_back(image_labels);
    }
}

void BoatDetector::negativeMining(std::vector<cv::Mat> training_images,
                                  std::vector<std::vector<cv::Rect>> positive_labels,
                                  std::vector<std::vector<cv::Rect>> &negative_labels) {
    cv::RNG rng;

    for (size_t i = 0; i < training_images.size(); i++) {
        int j = 0;
        int tries = 0;
        std::vector<cv::Rect> negative_image_labels;

        while (j < 5 & tries < 10) {
            cv::Point tl(rng.uniform(0, training_images[i].cols - 32),
                         rng.uniform(0, training_images[i].rows - 32));
            cv::Point br(rng.uniform(tl.x + 32, training_images[i].cols),
                         rng.uniform(tl.y + 32, training_images[i].rows));
            cv::Rect possible_negative(tl, br);
            if (possible_negative.area() < 5000) {
                tries++;
                continue;
            }

            bool done = true;
            for (cv::Rect positive : positive_labels[i]) {
                float overlapping_area = (positive & possible_negative).area();
                float union_area = positive.area() + possible_negative.area() - overlapping_area;

                if ((overlapping_area / union_area) > 0.25) {
                    done = false;
                    tries++;
                    continue;
                }
            }
            if (done) {
                negative_image_labels.push_back(possible_negative);
                j++;
                tries = 0;
            }
        }
        negative_labels.push_back(negative_image_labels);
    }
}

std::vector<cv::Rect> BoatDetector::detectBoats(cv::Mat input_img, int proposed_regions) {

    std::vector<cv::Rect> possible_boats = segmentationKDRP(input_img, proposed_regions);

    std::vector<cv::Rect> rects_boats;
    std::vector<float> probabilities;
    for (cv::Rect rect : possible_boats) {

        cv::Mat histogram;
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(input_img(rect), keypoints);
        if (keypoints.size() < 5) {
            continue;
        }
        bow->compute(input_img(rect), keypoints, histogram);
        if (histogram.rows != 0) {
            cv::Mat results;
            mlp->predict(histogram, results);
            if ((results.at<float>(0, 0) > 0.75) & (results.at<float>(0, 1) < 0.25)) {
                rects_boats.push_back(rect);
                probabilities.push_back(results.at<float>(0, 0));
            }
        }
    }

    return NMS(rects_boats, probabilities);
}

std::vector<cv::Rect> BoatDetector::NMS(std::vector<cv::Rect> ships,
                                        std::vector<float> probabilities) {
    std::vector<int> indeces;
    std::vector<cv::Rect> shipsGood;
    cv::sortIdx(probabilities, indeces, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
    for (size_t i = 0; i < ships.size(); i++) {
        int index_i = indeces[i];
        if (probabilities[index_i] == 0) {
            continue;
        }
        for (size_t j = i + 1; j < ships.size(); j++) {
            int index_j = indeces[j];
            if (probabilities[index_i] > probabilities[index_j]) {
                int intersection_area = (ships[index_i] & ships[index_j]).area();
                int union_area = ships[index_i].area() + ships[index_j].area() - intersection_area;
                if (intersection_area / union_area > threshold_IoU) {
                    probabilities[index_j] = 0;
                }
            }
        }
        shipsGood.push_back(ships[index_i]);
    }
    return shipsGood;
}

std::vector<cv::Rect> BoatDetector::segmentationKDRP(cv::Mat img, int proposed_regions) {
    std::vector<cv::Rect> proposals;
    proposals.reserve(proposed_regions);
    std::vector<cv::KeyPoint> keypoints_vector;
    cv::FAST(img, keypoints_vector, 10);

    cv::Mat bins_densities(17, 17, CV_32FC1, cv::Scalar(0.));
    cv::Mat points(img.size(), CV_32FC1, cv::Scalar(0.));
    float x_strife = img.cols / 17.;
    float y_strife = img.rows / 17.;
    float area_scale = 1. / (x_strife * y_strife * 4);
    for (cv::KeyPoint keypoint : keypoints_vector) {
        int x_bin = keypoint.pt.x / x_strife;
        int y_bin = keypoint.pt.y / y_strife;
        bins_densities.at<float>(y_bin, x_bin) =
            bins_densities.at<float>(y_bin, x_bin) + area_scale;
        points.at<float>(keypoint.pt) = points.at<float>(keypoint.pt) + 1.f;
    }

    cv::Mat keypoint_densities(16, 16, CV_32FC1, cv::Scalar(0.));
    for (size_t i = 0; i < keypoint_densities.rows; i++) {
        for (size_t j = 0; j < keypoint_densities.cols; j++) {
            keypoint_densities.at<float>(i, j) =
                bins_densities.at<float>(i, j) + bins_densities.at<float>(i + 1, j) +
                bins_densities.at<float>(i, j + 1) + bins_densities.at<float>(i + 1, j + 1);
        }
    }

    cv::Scalar mean, std_dev;
    cv::meanStdDev(keypoint_densities, mean, std_dev);

    cv::Mat integral_image;
    cv::integral(points, integral_image, CV_32FC1);
    int proposed_count = 0;
    cv::RNG rng;
    while (proposed_count < proposed_regions) {

        int tl_x = rng.uniform(0, img.cols - 32);
        int tl_y = rng.uniform(0, img.rows - 32);
        int br_x = rng.uniform(tl_x + 32, img.cols);
        int br_y = rng.uniform(tl_y + 32, img.rows);
        if (abs(1 - (br_x - tl_x) / (br_y - tl_y)) > 0.7) {
            continue;
        }

        float points_total = integral_image.at<float>(br_y + 1, br_x + 1) -
                             integral_image.at<float>(br_y + 1, tl_x) -
                             integral_image.at<float>(tl_y, br_x + 1) +
                             integral_image.at<float>(tl_y, tl_x);
        float area = (br_x - tl_x) * (br_y - tl_y);
        if (((points_total / area - mean.val[0]) / std_dev.val[0]) < rng.gaussian(1)) {
            continue;
        }
        proposals.push_back(cv::Rect(cv::Point(tl_x, tl_y), cv::Point(br_x, br_y)));
        proposed_count++;
    }

    return proposals;
}

float BoatDetector::testResult(std::vector<cv::Rect> found_boats,
                               std::vector<cv::Rect> gound_truths) {

    return 0;
}