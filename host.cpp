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
  // queue with best device
  queue q(getBestDevice());
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;
  
  {
    // scenario
    std::cout << "********** USM" << std::endl;
    
    // USM allocation and initialization
    double *data = malloc_shared<double>(N, q);
    for (int i = 0; i < N; i++) data[i] = 100000.0;
      
    // events
    std::vector<event> events_none;

    // call our kernels
    do_sqrt(q, N, B, data, nullptr, events_none);
    do_sqr(q, N, B, data, nullptr, events_none);
    do_add(5.0, q, N, B, data, nullptr, events_none).wait();
      
    // display results
    for (int i = 0; i < N; i++) std::cout << data[i] << " ";
    std::cout << std::endl;
    
    // free memory
    free(data, q);
  }
  
  {
    // scenario
    std::cout << "********** USM + events without waiting for done events" << std::endl;
    
    // USM allocation and initialization
    double *data = malloc_shared<double>(N, q);
    for (int i = 0; i < N; i++) data[i] = 100000.0;
      
    // events
    std::vector<event> events_none;
    std::vector<event> events_sqrt_sqr;
    std::vector<event> events_done;

    // call our kernels
    events_sqrt_sqr.push_back(do_sqrt(q, N, B, data, nullptr, events_none));
    events_sqrt_sqr.push_back(do_sqr(q, N, B, data, nullptr, events_none));
    events_done.push_back(do_add(5.0, q, N, B, data, nullptr, events_sqrt_sqr));
    
    // display results
    for (int i = 0; i < N; i++) std::cout << data[i] << " ";
    std::cout << std::endl;
    
    // free memory
    free(data, q);
  }

  {
    // scenario
    std::cout << "********** USM + events with waiting for done events" << std::endl;
    
    // USM allocation and initialization
    double *data = malloc_shared<double>(N, q);
    for (int i = 0; i < N; i++) data[i] = 100000.0;
      
    // events
    std::vector<event> events_none;
    std::vector<event> events_sqrt_sqr;
    std::vector<event> events_done;

    // call our kernels
    events_sqrt_sqr.push_back(do_sqrt(q, N, B, data, nullptr, events_none));
    events_sqrt_sqr.push_back(do_sqr(q, N, B, data, nullptr, events_none));
    events_done.push_back(do_add(5.0, q, N, B, data, nullptr, events_sqrt_sqr));
      
    // wait on all done events
    event done_waiter;
    done_waiter.wait(events_done);
    
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
      
    // events
    std::vector<event> events_none;

    // call our kernels
    do_sqrt(q, N, B, nullptr, &data, events_none);
    do_sqr(q, N, B, nullptr, &data, events_none);
    do_add(5.0, q, N, B, nullptr, &data, events_none).wait();
      
    // display results
    std::for_each(v.begin(), v.end(), [](const double& item) { std::cout << item << " "; });
    std::cout << std::endl;
  }
    
  {
    // scenario
    std::cout << "********** Buffer with Accessor + events without waiting for done events" << std::endl;
    
    // buffer allocation and initialization
    std::vector<double> v(N);
    std::generate(v.begin(), v.end(), [] () mutable { return 100000.0; });
    buffer data(v);
      
    // events
    std::vector<event> events_none;
    std::vector<event> events_sqrt_sqr;
    std::vector<event> events_done;

    // call our kernels
    events_sqrt_sqr.push_back(do_sqrt(q, N, B, nullptr, &data, events_none));
    events_sqrt_sqr.push_back(do_sqr(q, N, B, nullptr, &data, events_none));
    events_done.push_back(do_add(5.0, q, N, B, nullptr, &data, events_sqrt_sqr));
      
    // display results
    std::for_each(v.begin(), v.end(), [](const double& item) { std::cout << item << " "; });
    std::cout << std::endl;
  }

  {
    // scenario
    std::cout << "********** Buffer with Accessor + events with explicit waiting for done events" << std::endl;
    
    // buffer allocation and initialization
    std::vector<double> v(N);
    std::generate(v.begin(), v.end(), [] () mutable { return 100000.0; });
    buffer data(v);
      
    // events
    std::vector<event> events_none;
    std::vector<event> events_sqrt_sqr;
    std::vector<event> events_done;

    // call our kernels
    events_sqrt_sqr.push_back(do_sqrt(q, N, B, nullptr, &data, events_none));
    events_sqrt_sqr.push_back(do_sqr(q, N, B, nullptr, &data, events_none));
    events_done.push_back(do_add(5.0, q, N, B, nullptr, &data, events_sqrt_sqr));
      
    // wait on all done events
    event done_waiter;
    done_waiter.wait(events_done);

    // display results
    std::for_each(v.begin(), v.end(), [](const double& item) { std::cout << item << " "; });
    std::cout << std::endl;
  }

  {
    // scenario
    std::cout << "********** Buffer with Accessor + events with implicit waiting for done events via host accessor" << std::endl;
    
    // buffer allocation and initialization
    std::vector<double> v(N);
    std::generate(v.begin(), v.end(), [] () mutable { return 100000.0; });
    buffer data(v);
      
    // events
    std::vector<event> events_none;
    std::vector<event> events_sqrt_sqr;
    std::vector<event> events_done;

    // call our kernels
    events_sqrt_sqr.push_back(do_sqrt(q, N, B, nullptr, &data, events_none));
    events_sqrt_sqr.push_back(do_sqr(q, N, B, nullptr, &data, events_none));
    events_done.push_back(do_add(5.0, q, N, B, nullptr, &data, events_sqrt_sqr));
        
    // display results
    host_accessor result{data};
    for (int i = 0; i < result.get_count(); i++) std::cout << result[i] << " ";
    std::cout << std::endl;
  }

  return 0;
}
