
cmake_minimum_required (VERSION 3.8)

project (lab1 LANGUAGES CXX)

include_directories(${CMAKE_SOURCE_DIR}/../../../opencv/include)
link_directories(${CMAKE_SOURCE_DIR}/../../../opencv/x64/vc16/lib)

add_executable (lab1 "main.cpp")
target_link_libraries(lab1 opencv_world452d)