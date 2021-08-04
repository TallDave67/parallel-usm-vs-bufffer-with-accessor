cmake_minimum_required(VERSION 3.16)
project(usm-vs-bufffer-with-accessor-device)

set(CMAKE_VERBOSE_MAKEFILE ON)


add_library(usm-vs-bufffer-with-accessor-device
  device_factory.cpp
)

target_compile_options(usm-vs-bufffer-with-accessor-device PRIVATE 
  -std=c++20 -stdlib=libstdc++
)

