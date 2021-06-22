#pragma once

#include <string>
#include <vector>

namespace sea_segmentation {
void prepare_dataset(std::vector<std::string> arguments);
void segment(std::vector<std::string> arguments);
} // namespace sea_segmentation