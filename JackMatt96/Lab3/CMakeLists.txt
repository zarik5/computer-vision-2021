cmake_minimum_required (VERSION 3.8)

project ("Lab3")

find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/main.cpp include/filter.h src/filter.cpp src/show_histogram_function.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

