#include "sea_segmentation.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

namespace fs = std::filesystem;

// For computing the final CNN model, the train split is set to contain almost all dataset images
const float TRAIN_SPLIT_PROPORTION = 0.95;

const int FOCUS_WIN_SIDE = 9;
const int CONTEXT_WIN_SIDE = 32;
const int CONTEXT_LEVELS_COUNT = 4;

// Pixel values for the classes in the ADE20K dataset
const int WATER_CLASS = 22;
const int SEA_CLASS = 27;

// controls the size of the dataset for the classifier
const int SAMPLES_PER_REGION = 10;

// images are reduced to this size for performcance purpose
const int MAX_IMAGE_WIDTH = 400;

const std::string PREPROC_HELP_MESSAGE = R"help(
sea_train: Extract square windows of pixels and their class (sea or non-sea), then train a classifier

USAGE:
final_project sea_train <dataset_dir> <out_dir>
Note: the dataset must be ADE20K Outdoors.
)help";

const std::string SEGMENT_HELP_MESSAGE = R"help(
sea_segment: Segment the sea from an image

USAGE:
final_project sea_segment <model> <in_image> <out_dir> <target_segmentation>
)help";

cv::Mat resize_image(cv::Mat image) {
    if (image.cols > MAX_IMAGE_WIDTH) {
        cv::Mat output;
        int height = image.rows * MAX_IMAGE_WIDTH / image.cols;
        cv::resize(image, output, {MAX_IMAGE_WIDTH, height});

        return output;
    } else {
        return image;
    }
}

cv::Mat get_biggest_blob_with_holes(cv::Mat image) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    float biggest_area = 0;
    std::vector<cv::Point> biggest_contour;
    std::vector<std::vector<cv::Point>> other_contours;
    for (int i = 0; i < contours.size(); i++) {
        if (hierarchy[i][3] == -1) {
            double area = cv::contourArea(contours[i], false);
            if (area > biggest_area) {
                biggest_area = area;
                biggest_contour = contours[i];
            }
        } else {
            other_contours.push_back(contours[i]);
        }
    }

    auto biggest_contour_vec = std::vector<std::vector<cv::Point>>({biggest_contour});
    cv::Mat output = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::fillPoly(output, biggest_contour_vec, 255);
    cv::fillPoly(output, other_contours, 0);

    return output;
}

struct Windows {
    cv::Mat focus;
    std::vector<cv::Mat> context;
};

class ScalesGenerator {
  private:
    std::vector<cv::Mat> pyramid;

    // Fill with 0s if `roi` goes outside `source` boundary.
    cv::Mat get_region(cv::Mat &source, cv::Rect roi) {
        auto source_roi = cv::Rect({0, 0}, source.size());
        auto intersection = source_roi & roi;
        auto offset_roi = intersection - roi.tl();

        auto out_region = (cv::Mat)cv::Mat::zeros(roi.size(), source.type());
        source(intersection).copyTo(out_region(offset_roi));

        return out_region;
    }

  public:
    ScalesGenerator(cv::Mat image) {
        this->pyramid.push_back(image);
        for (int i = 0; i < CONTEXT_LEVELS_COUNT - 1; i++) {
            cv::Mat scaled_image;
            cv::pyrDown(image, scaled_image);
            this->pyramid.push_back(scaled_image);
            image = scaled_image;
        }
    }

    Windows extract_window_scales_for_pixel(cv::Point point) {
        const auto FOCUS_WIN_SIZE = cv::Point(FOCUS_WIN_SIDE, FOCUS_WIN_SIDE);
        const auto CONTEXT_WIN_SIZE = cv::Point(CONTEXT_WIN_SIDE, CONTEXT_WIN_SIDE);

        auto focus_roi = cv::Rect(point - FOCUS_WIN_SIZE / 2, cv::Size(FOCUS_WIN_SIZE));
        auto focus_window = this->get_region(this->pyramid[0], focus_roi);

        auto context_windows = std::vector<cv::Mat>();
        for (int i = 0; i < CONTEXT_LEVELS_COUNT; i++) {
            auto context_roi =
                cv::Rect(point / std::pow(2, i) - CONTEXT_WIN_SIZE / 2, cv::Size(CONTEXT_WIN_SIZE));
            context_windows.push_back(this->get_region(this->pyramid[i], context_roi));
        }

        return {focus_window, context_windows};
    }
};

struct DatasetClassSplit {
    std::vector<std::string> with_sea_names;
    std::vector<std::string> without_sea_names;
};

static DatasetClassSplit split_dataset_by_class(fs::path dataset_dir) {
    std::vector<std::string> annotations_paths;
    cv::glob(dataset_dir / "annotations/training/*.png", annotations_paths);

    auto split = DatasetClassSplit();
    for (auto &path : annotations_paths) {
        auto annot = cv::imread(path, cv::IMREAD_GRAYSCALE);

        bool has_sea = false;
        for (int x = 0; x < annot.cols; x++) {
            for (int y = 0; y < annot.rows; y++) {
                auto value = annot.at<uchar>(y, x);
                if (value == WATER_CLASS || value == SEA_CLASS) {
                    has_sea = true;
                    break;
                }
            }
            if (has_sea) {
                break;
            }
        }

        auto name_start_idx = path.rfind("/") + 1;
        auto name_len = path.size() - name_start_idx - 4;
        auto name = path.substr(name_start_idx, name_len);

        if (has_sea) {
            split.with_sea_names.push_back(name);
        } else {
            split.without_sea_names.push_back(name);
        }
    }

    return split;
}

static std::string extract_windows_and_save(ScalesGenerator &generator, fs::path &out_dir,
                                            std::string &image_name, cv::Point point,
                                            int point_index) {
    auto [focus, context] = generator.extract_window_scales_for_pixel(point);

    auto sample_name = image_name + "_" + std::to_string(point_index);
    auto path_prefix = (out_dir / sample_name).string();

    cv::imwrite(path_prefix + "_focus.png", focus);
    for (int i = 0; i < context.size(); i++) {
        cv::imwrite(path_prefix + "_context_" + std::to_string(i) + ".png", context[i]);
    }

    return sample_name;
}

struct SamplingParams {
    fs::path dataset_dir;
    fs::path out_dir;
    DatasetClassSplit names;
    std::ofstream train_classes_file;
    std::ofstream test_classes_file;
};

static void sample_images_with_sea(SamplingParams &params) {
    const int MORPH_ITERS = FOCUS_WIN_SIDE / 2 - 1;

    auto &[dataset_dir, out_dir, names, train_classes_file, test_classes_file] = params;
    auto [with_sea_names, without_sea_names] = names;

    auto random_engine = std::mt19937();

    float train_count = with_sea_names.size() * TRAIN_SPLIT_PROPORTION;

    for (int i = 0; i < with_sea_names.size(); i++) {
        std::cout << "Sampling images with sea... " << i + 1 << "/" << with_sea_names.size()
                  << std::endl;

        auto name = with_sea_names[i];

        auto annot = cv::imread(dataset_dir / "annotations/training" / (name + ".png"),
                                cv::IMREAD_GRAYSCALE);

        // Find sea, edge and non-sea pixels locations. Edge points refers to pixels near the edge
        // between sea and non-sea regions.

        cv::Mat sea_mask;
        cv::inRange(annot, SEA_CLASS, SEA_CLASS, sea_mask);
        cv::Mat water_mask;
        cv::inRange(annot, WATER_CLASS, WATER_CLASS, water_mask);
        sea_mask += water_mask;

        cv::Mat eroded_sea_mask;
        cv::erode(sea_mask, eroded_sea_mask, cv::noArray(), {-1, -1}, MORPH_ITERS);

        cv::Mat dilated_sea_mask;
        cv::dilate(sea_mask, dilated_sea_mask, cv::noArray(), {-1, -1}, MORPH_ITERS);

        auto sea_points = std::vector<cv::Point>();
        auto inner_edge_points = std::vector<cv::Point>();
        auto outer_edge_points = std::vector<cv::Point>();
        auto non_sea_points = std::vector<cv::Point>();
        for (int y = 0; y < annot.rows; y++) {
            for (int x = 0; x < annot.cols; x++) {
                bool is_eroded_sea = eroded_sea_mask.at<uint8_t>(y, x) == 255;
                bool is_sea = sea_mask.at<uint8_t>(y, x) == 255;
                bool is_dilated_sea = dilated_sea_mask.at<uint8_t>(y, x) == 255;
                if (is_eroded_sea && is_sea && is_dilated_sea) {
                    sea_points.push_back({x, y});
                } else if (is_sea && is_dilated_sea) {
                    inner_edge_points.push_back({x, y});
                } else if (is_dilated_sea) {
                    outer_edge_points.push_back({x, y});
                } else {
                    non_sea_points.push_back({x, y});
                }
            }
        }

        auto image = cv::imread(dataset_dir / "images/training" / (name + ".jpg"));
        auto generator = ScalesGenerator(image);

        auto sampled_sea_points = std::vector<cv::Point>();
        auto sampled_non_sea_points = std::vector<cv::Point>();

        std::sample(sea_points.begin(), sea_points.end(), std::back_inserter(sampled_sea_points),
                    SAMPLES_PER_REGION, random_engine);
        std::sample(inner_edge_points.begin(), inner_edge_points.end(),
                    std::back_inserter(sampled_sea_points), SAMPLES_PER_REGION / 2, random_engine);
        std::sample(outer_edge_points.begin(), outer_edge_points.end(),
                    std::back_inserter(sampled_non_sea_points), SAMPLES_PER_REGION / 2,
                    random_engine);
        std::sample(non_sea_points.begin(), non_sea_points.end(),
                    std::back_inserter(sampled_non_sea_points), SAMPLES_PER_REGION / 2,
                    random_engine);

        auto out_dir_for_split = out_dir / (i < train_count ? "train" : "test");
        auto &classes_file = (i < train_count ? train_classes_file : test_classes_file);

        int index = 0;
        for (auto pt : sampled_sea_points) {
            auto sample_name =
                extract_windows_and_save(generator, out_dir_for_split, name, pt, index);
            classes_file << sample_name + " 1" << std::endl;

            index++;
        }

        for (auto pt : sampled_non_sea_points) {
            auto sample_name =
                extract_windows_and_save(generator, out_dir_for_split, name, pt, index);
            classes_file << sample_name + " 0" << std::endl;

            index++;
        }
    }
}

static void sample_images_without_sea(SamplingParams &params) {
    auto &[dataset_dir, out_dir, names, train_classes_file, test_classes_file] = params;
    auto [with_sea_names, without_sea_names] = names;

    // with_sea_count = without_sea_count * sample_prob
    auto sample_prob = static_cast<float>(with_sea_names.size()) / without_sea_names.size();

    auto rng = cv::RNG();

    float train_count = without_sea_names.size() * TRAIN_SPLIT_PROPORTION;

    for (int i = 0; i < without_sea_names.size(); i++) {
        std::cout << "Sampling images without sea... " << i + 1 << "/" << without_sea_names.size()
                  << std::endl;

        auto name = without_sea_names[i];

        auto image = cv::imread(dataset_dir / "images/training" / (name + ".jpg"));
        auto generator = ScalesGenerator(image);

        auto out_dir_for_split = out_dir / (i < train_count ? "train" : "test");
        auto &classes_file = (i < train_count ? train_classes_file : test_classes_file);

        int index = 0;
        for (int j = 0; j < SAMPLES_PER_REGION / 2; j++) {
            if (rng.uniform(0.f, 1.f) > sample_prob) {
                auto point = cv::Point(rng(image.cols), rng(image.rows));

                auto sample_name =
                    extract_windows_and_save(generator, out_dir_for_split, name, point, index);
                classes_file << sample_name + " 0" << std::endl;

                index++;
            }
        }
    }
}

static std::vector<float> non_contiguous_mat_to_float_vector(cv::Mat mat) {
    auto data = std::vector<float>();
    for (int i = 0; i < mat.rows; i++) {
        data.insert(data.end(), mat.ptr<float>(i), mat.ptr<float>(i) + mat.cols * 3);
    }

    return data;
}

// Divide dataset into small windows of pixels used for training a classifier
static void prepare_dataset(fs::path dataset_dir, fs::path out_dir) {
    std::filesystem::create_directories(out_dir / "train");
    std::filesystem::create_directories(out_dir / "test");

    std::cout << "Find which images have a \"sea\" or \"water\" class..." << std::endl;
    auto names = split_dataset_by_class(dataset_dir);

    auto train_classes_file = std::ofstream();
    train_classes_file.open(out_dir / "train/classes.txt");

    auto test_classes_file = std::ofstream();
    test_classes_file.open(out_dir / "test/classes.txt");

    SamplingParams params = {dataset_dir, out_dir, names, std::move(train_classes_file),
                             std::move(test_classes_file)};

    // Construct dataset with balanced splits: 1/3 sea, 1/6 inner edge, 1/6 outer edge, 1/6
    // non-sea within sea image, 1/6 non-sea within non-sea image

    sample_images_with_sea(params);

    sample_images_without_sea(params);
}

namespace sea_segmentation {
void train(std::vector<std::string> arguments) {
    if (arguments.size() != 2) {
        std::cout << PREPROC_HELP_MESSAGE;
        return;
    }

    auto dataset_dir = fs::path(arguments[0]);
    auto out_dir = fs::path(arguments[1]);

    prepare_dataset(dataset_dir, out_dir);

    // Invoke Python script for the training
    auto command = std::ostringstream();
    command << "python3 ../sea_train.py " << FOCUS_WIN_SIDE << " " << CONTEXT_WIN_SIDE << " "
            << CONTEXT_LEVELS_COUNT << " " << dataset_dir << " " << out_dir;
    system(command.str().c_str());
}

void segment_image(std::vector<std::string> arguments) {
    if (arguments.size() != 4) {
        std::cout << SEGMENT_HELP_MESSAGE;
        return;
    }

    auto model_path = fs::path(arguments[0]);
    auto image_path = fs::path(arguments[1]);
    auto out_dir = fs::path(arguments[2]);
    auto target_segmentation_path = fs::path(arguments[3]);

    fs::create_directories(out_dir);

    auto model = cv::dnn::readNet(model_path);

    auto image = cv::imread(image_path);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3, 1.0 / 255);
    image = resize_image(image);

    auto generator = ScalesGenerator(image);

    auto segmentation = cv::Mat(image.rows, image.cols, CV_32F);

    for (int y = 0; y < image.rows; y++) {
        std::cout << "Progress: " << y * 100 / image.rows << "%" << std::endl;

        for (int x = 0; x < image.cols; x++) {
            auto [focus, context] = generator.extract_window_scales_for_pixel({x, y});

            model.setInput(cv::dnn::blobFromImage(focus), "x");
            for (int i = 0; i < CONTEXT_LEVELS_COUNT; i++) {
                model.setInput(cv::dnn::blobFromImage(context[i]), "x_" + std::to_string(i + 1));
            }

            segmentation.at<float>(y, x) = model.forward().at<float>(0);
        }
    }

    cv::imshow("segmentation raw", segmentation);

    cv::threshold(segmentation, segmentation, 0.5, 1, cv::THRESH_BINARY);

    segmentation = segmentation * 255;
    segmentation.convertTo(segmentation, CV_8UC1);

    auto kernel = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
    cv::erode(segmentation, segmentation, kernel);
    segmentation = get_biggest_blob_with_holes(segmentation);
    cv::dilate(segmentation, segmentation, kernel);

    cv::imshow("segmentation", segmentation);

    auto target_segmentation = cv::imread(target_segmentation_path, cv::IMREAD_GRAYSCALE);
    target_segmentation = resize_image(target_segmentation);

    auto pixel_accuracy = (1 - (float)cv::sum(cv::abs(target_segmentation - segmentation))[0] /
                                   (image.rows * image.cols * 255));

    std::cout << "Pixel accuracy: " << pixel_accuracy << std::endl;

    auto channels = std::vector<cv::Mat>(
        {cv::Mat::zeros(image.rows, image.cols, CV_8UC1), target_segmentation, segmentation});
    cv::Mat evaluation_image;
    cv::merge(channels, evaluation_image);

    cv::imshow("evaluation", evaluation_image);
    cv::waitKey();

    cv::imwrite(out_dir / (image_path.stem().string() + "_segmentation.png"), segmentation);
    cv::imwrite(out_dir / (image_path.stem().string() + "_evaluation.png"), evaluation_image);
}

} // namespace sea_segmentation