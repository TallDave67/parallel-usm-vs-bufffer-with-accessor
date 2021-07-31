// This code by David Bellis in 2021 (inspired by Intel sample code)

#include <CL/sycl.hpp>
using namespace sycl;

#include "do_add.h"
#include "do_sqr.h"
#include "do_sqrt.h"

static const unsigned long N = 2048;
static const unsigned long B = 4;

int main() {
  queue q(cpu_selector{});
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;

  // USM allocation and initialization
  double *p_data = malloc_shared<double>(N, q);
  for (int i = 0; i < N; i++) p_data[i] = 100000.0;
    
  // call our kernels
  do_sqrt(q, N, B, p_data, nullptr, nullptr);
  do_sqr(q, N, B, p_data, nullptr, nullptr);
  do_add(5.0, q, N, B, p_data, nullptr, nullptr).wait();
    
  // display results
  for (int i = 0; i < N; i++) std::cout << p_data[i] << " ";
  std::cout << std::endl;
  
  // free memory
  free(p_data, q);
  
  return 0;
}
