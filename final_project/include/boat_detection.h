#pragma once

#include <string>
#include <vector>

namespace boat_detection {
void train(std::vector<std::string> arguments);
void detect(std::vector<std::string> arguments);
} // namespace boat_detection