
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project (lab4 LANGUAGES CXX)

include_directories(
    ${CMAKE_SOURCE_DIR}/../../../opencv/include
    ./
)
link_directories(${CMAKE_SOURCE_DIR}/../../../opencv/x64/vc16/lib)

add_executable (lab4 "main.cpp" "lines_detection.cpp" "circle_detection.cpp")
target_link_libraries(lab4 opencv_world452d)