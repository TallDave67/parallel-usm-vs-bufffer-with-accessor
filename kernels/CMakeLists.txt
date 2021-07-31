cmake_minimum_required(VERSION 3.16)
project(usm-vs-bufffer-with-accessor-kernels)

set(CMAKE_VERBOSE_MAKEFILE ON)


add_library(usm-vs-bufffer-with-accessor-kernels
  do_add.cpp
  do_sqr.cpp
  do_sqrt.cpp
)

target_compile_options(usm-vs-bufffer-with-accessor-kernels PRIVATE 
  -std=c++20 -stdlib=libstdc++
)

