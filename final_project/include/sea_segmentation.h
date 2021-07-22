#pragma once

#include <string>
#include <vector>

namespace sea_segmentation {
void prepare_dataset(std::vector<std::string> arguments);
void segment_image(std::vector<std::string> arguments);
void show_segmentation(std::vector<std::string> arguments);
} // namespace sea_segmentation