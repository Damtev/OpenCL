cmake_minimum_required(VERSION 3.15)
project(openCl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

include_directories(.)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable(openCl main.cpp)
target_link_libraries (openCl ${OpenCL_LIBRARY})