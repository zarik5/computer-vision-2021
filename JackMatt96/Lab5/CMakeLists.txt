cmake_minimum_required (VERSION 3.8)

project ("Lab5")
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} src/lab5.cpp include/panoramic_utils.h include/PanoramicImage.h src/PanoramicImage.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

