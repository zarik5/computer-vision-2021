cmake_minimum_required (VERSION 2.8)

project ("Lab6")
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} src/lab6.cpp include/VideoObjectTracking.h src/VideoObjectTracking.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")