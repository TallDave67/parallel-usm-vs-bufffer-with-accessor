// This code by David Bellis in 2021 (inspired by Intel sample code)

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
using namespace sycl;

#include "device_factory.h"
#include "do_add.h"
#include "do_sqr.h"
#include "do_sqrt.h"

static const unsigned long N = 2048;
static const unsigned long B = 4;

int main() {
  queue q(getBestDevice());
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;

  {
    // scenario
    std::cout << "********** USM" << std::endl;
    
    // USM allocation and initialization
    double *data = malloc_shared<double>(N, q);
    for (int i = 0; i < N; i++) data[i] = 100000.0;
      
    // call our kernels
    do_sqrt(q, N, B, data, nullptr, nullptr);
    do_sqr(q, N, B, data, nullptr, nullptr);
    do_add(5.0, q, N, B, data, nullptr, nullptr).wait();
      
    // display results
    for (int i = 0; i < N; i++) std::cout << data[i] << " ";
    std::cout << std::endl;
    
    // free memory
    free(data, q);
  }
  
  {
    // scenario
    std::cout << "********** Buffer with Accessor" << std::endl;
    
    // buffer allocation and initialization
    std::vector<double> v(N);
    std::generate(v.begin(), v.end(), [] () mutable { return 100000.0; });
    buffer data(v);
      
    // call our kernels
    do_sqrt(q, N, B, nullptr, &data, nullptr);
    do_sqr(q, N, B, nullptr, &data, nullptr);
    do_add(5.0, q, N, B, nullptr, &data, nullptr).wait();
      
    // display results
    std::for_each(v.begin(), v.end(), [](const int& item) { std::cout << item << " "; });
    std::cout << std::endl;
  }
    
  return 0;
}
