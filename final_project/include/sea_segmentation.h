#pragma once

#include <string>
#include <vector>

namespace sea_segmentation {

// Divide dataset into small windows of pixels and train a classifier using Keras
void train(std::vector<std::string> arguments);

// Segment the sea in an image and compare the result with the provided ground truth
void segment_image(std::vector<std::string> arguments);

} // namespace sea_segmentation