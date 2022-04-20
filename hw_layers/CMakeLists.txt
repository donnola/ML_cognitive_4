cmake_minimum_required(VERSION 3.22)
project(hw_layers)

set(CMAKE_CXX_STANDARD 20)

find_package(jsoncpp REQUIRED)
add_executable(hw_layers main.cpp)
target_link_libraries(hw_layers jsoncpp_lib)