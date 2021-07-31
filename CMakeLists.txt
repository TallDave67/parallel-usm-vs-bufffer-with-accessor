cmake_minimum_required(VERSION 3.16)
project(usm-vs-bufffer-with-accessor)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(kernels)
include_directories(
  kernels
)

add_executable(usm-vs-bufffer-with-accessor
  host.cpp
)

target_compile_options(usm-vs-bufffer-with-accessor PRIVATE 
  -std=c++20 -stdlib=libstdc++
)

target_link_libraries(usm-vs-bufffer-with-accessor usm-vs-bufffer-with-accessor-kernels)
