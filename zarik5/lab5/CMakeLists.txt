
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project (lab5 LANGUAGES CXX)

include_directories(
    ${CMAKE_SOURCE_DIR}/../../../opencv/include
    ./
)
link_directories(${CMAKE_SOURCE_DIR}/../../../opencv/x64/vc16/lib)

add_executable (lab5 "main.cpp" "panoramic_image.cpp")
target_link_libraries(lab5 opencv_world452d)