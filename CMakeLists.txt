cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(lab1 LANGUAGES CXX)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lab1 src/main.cpp)

target_link_libraries(lab1 ${OpenCV_LIBS})

message(${OpenCV_INCLUDE_DIRS})