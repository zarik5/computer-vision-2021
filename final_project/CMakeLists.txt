
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project(final_project LANGUAGES CXX)

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} src/main.cpp src/boat_detection.cpp src/sea_segmentation.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} Eigen3::Eigen nlohmann_json::nlohmann_json)