// This code by David Bellis in 2021 (inspired by Intel sample code)

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
using namespace sycl;

#include <iostream>
#include <utility>

device && getBestDevice()
{
  // fpga is the #1 desired device
  try
  {
    INTEL::fpga_selector selector;
    queue q(selector);
    std::cout << "fpga is best device" << std::endl;
    device d = device(INTEL::fpga_selector());
    return std::move(d);
  }
  catch(sycl::exception& e)
  {
    std::cout << "no fpga device available" << std::endl;
  }

  // fpga emulator is the #2 desired device
  try
  {
    INTEL::fpga_emulator_selector selector;
    queue q(selector);
    std::cout << "fpga emulator is best device" << std::endl;
    device d = device(INTEL::fpga_emulator_selector());
    return std::move(d);
  }
  catch(sycl::exception& e)
  {
    std::cout << "no fpga emulator device available" << std::endl;
  }

  // gpu is the #3 desired device
  try
  {
    gpu_selector selector;
    queue q(selector);
    std::cout << "gpu is best device" << std::endl;
    device d = device(gpu_selector());
    return std::move(d);
  }
  catch(sycl::exception& e)
  {
    std::cout << "no gpu device available" << std::endl;
  }

  // cpu is the #4 desired device
  try
  {
    cpu_selector selector;
    queue q(selector);
    std::cout << "cpu is best device" << std::endl;
    device d = device(cpu_selector());
    return std::move(d);
  }
  catch(sycl::exception& e)
  {
    std::cout << "no cpu device available" << std::endl;
  }
}

