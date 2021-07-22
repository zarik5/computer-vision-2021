#include "sea_segmentation.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

const float TRAIN_SPLIT_PROPORTION = 0.9;

const int FOCUS_WIN_SIDE = 9;
const int SCALES_COUNT = 4;
const int CONTEXT_WIN_SIDE = 32;

const int WATER_CLASS = 22;
const int SEA_CLASS = 27;

const int SAMPLES_PER_REGION = 10;

const std::string PREPROC_HELP_MESSAGE = R"help(
sea_prep_data: Extract square windows of pixels and their class (sea or non-sea)

USAGE:
final_project sea_prep_data <in_dataset_dir> <out_dir>
Note: the dataset must be ADE20K Outdoors.
)help";

const std::string SEGMENT_HELP_MESSAGE = R"help(
sea_segment: Segment the sea from an image

USAGE:
final_project sea_segment <in_model_file> <in_image_file> <out_segmentation_file> [--display]

FLAG:
display: Show the segmentation result in a window before closing the program
)help";

const int MAX_IMAGE_WIDTH = 300;

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

struct Windows {
    cv::Mat focus;
    std::vector<cv::Mat> context;
};

class ScalesGenerator {
  private:
    std::vector<cv::Mat> pyramid;

    // Fill with 0s if `roi` goes outside `source` boundary.
    cv::Mat get_region(cv::Mat &source, cv::Rect roi) {
        auto source_roi = cv::Rect({}, source.size());
        auto intersection = source_roi & roi;
        auto offset_roi = intersection - roi.tl();
        // std::cout << source.size() << "\n" << roi << "\n" << offset_roi << std::endl;

        auto out_region = (cv::Mat)cv::Mat::zeros(roi.size(), source.type());
        source(intersection).copyTo(out_region(offset_roi));

        return out_region;
    }

  public:
    ScalesGenerator(cv::Mat image) {
        this->pyramid.push_back(image);
        for (int i = 0; i < SCALES_COUNT - 1; i++) {
            cv::Mat scaled_image;
            cv::pyrDown(image, scaled_image);
            this->pyramid.push_back(scaled_image);
            image = scaled_image;
        }
    }

    Windows extract_window_scales_for_pixel(cv::Point point) {
        // std::cout << point << std::endl;

        const auto FOCUS_WIN_SIZE = cv::Point(FOCUS_WIN_SIDE, FOCUS_WIN_SIDE);
        const auto CONTEXT_WIN_SIZE = cv::Point(CONTEXT_WIN_SIDE, CONTEXT_WIN_SIDE);

        auto focus_roi = cv::Rect(point - FOCUS_WIN_SIZE / 2, cv::Size(FOCUS_WIN_SIZE));
        auto focus_window = this->get_region(this->pyramid[0], focus_roi);

        auto context_windows = std::vector<cv::Mat>();
        for (int i = 0; i < SCALES_COUNT; i++) {
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

static DatasetClassSplit split_dataset_by_class(std::string dataset_dir) {
    std::vector<std::string> annotations_paths;
    cv::glob(dataset_dir + "/annotations/training/*.png", annotations_paths);

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

static std::string extract_windows_and_save(ScalesGenerator &generator, std::string &out_dir,
                                            std::string &image_name, cv::Point point,
                                            int point_index) {
    auto [focus, context] = generator.extract_window_scales_for_pixel(point);

    auto sample_name = image_name + "_" + std::to_string(point_index);
    auto path_prefix = out_dir + "/" + sample_name + "_";

    cv::imwrite(path_prefix + "focus.png", focus);
    for (int i = 0; i < context.size(); i++) {
        cv::imwrite(path_prefix + "context_" + std::to_string(i) + ".png", context[i]);
    }

    return sample_name;
}

struct SamplingParams {
    std::string dataset_dir;
    std::string out_dir;
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

        auto annot = cv::imread(dataset_dir + "/annotations/training/" + name + ".png",
                                cv::IMREAD_GRAYSCALE);

        // Find sea, edge and non-sea pixels locations. Edge points refers to pixels near the edge
        // between sea and non-sea classes.

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

        auto image = cv::imread(dataset_dir + "/images/training/" + name + ".jpg");
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

        auto out_dir_for_split = out_dir + (i < train_count ? "/train" : "/test");
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

        auto image = cv::imread(dataset_dir + "/images/training/" + name + ".jpg");
        auto generator = ScalesGenerator(image);

        auto out_dir_for_split = out_dir + (i < train_count ? "/train" : "/test");
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

namespace sea_segmentation {
void prepare_dataset(std::vector<std::string> arguments) {
    if (arguments.size() != 2) {
        std::cout << PREPROC_HELP_MESSAGE;
        return;
    }

    auto dataset_dir = arguments[0];
    auto out_dir = arguments[1];

    std::filesystem::create_directories(out_dir + "/train");
    std::filesystem::create_directories(out_dir + "/test");

    std::cout << "Find which images have a \"sea\" or \"water\" class..." << std::endl;
    auto names = split_dataset_by_class(dataset_dir);

    auto train_classes_file = std::ofstream();
    train_classes_file.open(out_dir + "/train/classes.txt");

    auto test_classes_file = std::ofstream();
    test_classes_file.open(out_dir + "/test/classes.txt");

    SamplingParams params = {dataset_dir, out_dir, names, std::move(train_classes_file),
                             std::move(test_classes_file)};

    // Construct dataset with balanced splits: 1/3 sea, 1/6 inner edge, 1/6 outer edge, 1/6
    // non-sea within sea image, 1/6 non-sea within non-sea image

    sample_images_with_sea(params);

    sample_images_without_sea(params);
}

void segment_image(std::vector<std::string> arguments) {
    // if (arguments.size() != 2) {
    //     std::cout << SEGMENT_HELP_MESSAGE;
    //     return;
    // }

    // auto image_file = arguments[0];
    // auto target_segmentation_file = arguments[1];
    // auto output_dir = arguments[2];
    auto model_file = "../out/model.pb";
    auto image_file = "../data/Kaggle_ships/aida-ship-driving-cruise-ship-sea-144796.jpg";
    auto target_segmentation_file =
        "../data/Kaggle_ships/aida-ship-driving-cruise-ship-sea-144796.jpg";
    auto output_dir = std::string("../out");

    auto fdjksf = cv::dnn::readNet(model_file);

    std::exit(0);

    auto image = cv::imread(image_file);
    image = resize_image(image);

    std::filesystem::create_directories(output_dir);

    auto dimensions_file = std::ofstream(output_dir + "/dimensions.txt");
    dimensions_file << image.cols << std::endl << image.rows << std::endl;
    dimensions_file.close();

    auto generator = ScalesGenerator(image);

    for (int y = 0; y < image.rows; y++) {
        std::cout << "Progress: " << y * 100 / image.rows << "%" << std::endl;

        for (int x = 0; x < image.cols; x++) {
            auto base_path = output_dir + "/" + std::to_string(y * image.cols + x);

            auto [focus, context] = generator.extract_window_scales_for_pixel({x, y});

            cv::imwrite(base_path + "_focus.png", focus);

            for (int i = 0; i < SCALES_COUNT; i++) {
                cv::imwrite(base_path + "_context_" + std::to_string(i) + ".png", context[i]);
            }
        }
    }
}

void show_segmentation(std::vector<std::string> arguments) {
    if (arguments.size() != 1) {
        std::cout << SEGMENT_HELP_MESSAGE;
        return;
    }

    auto segmentation_dir = arguments[0];

    auto dimensions_file = std::ifstream(segmentation_dir + "/dimensions.txt");
    int width, height;
    dimensions_file >> width >> height;

    auto pixels_count = width * height;

    auto segmentation_file =
        std::ifstream(segmentation_dir + "/segmentation", std::ios_base::binary);
    auto buffer = std::vector<float>(pixels_count);
    segmentation_file.read((char *)&buffer[0], pixels_count * 4);

    auto image = cv::Mat(height, width, CV_32FC1, &buffer[0]);

    cv::imshow("segmentation", image);
    cv::waitKey();

    cv::threshold(image, image, 0.5, 1, cv::THRESH_BINARY);

    cv::imshow("segmentation", image);
    cv::waitKey();
}

} // namespace sea_segmentation