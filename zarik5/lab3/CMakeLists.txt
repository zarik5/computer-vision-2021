
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project (lab3 LANGUAGES CXX)

include_directories(
    ${CMAKE_SOURCE_DIR}/../../../opencv/include
    ./
)
link_directories(${CMAKE_SOURCE_DIR}/../../../opencv/x64/vc16/lib)

add_executable (lab3 "main.cpp" "histogram_tests.cpp" "denoise_tests.cpp" "filter.cpp" "morphology_tests.cpp")
target_link_libraries(lab3 opencv_world452d)