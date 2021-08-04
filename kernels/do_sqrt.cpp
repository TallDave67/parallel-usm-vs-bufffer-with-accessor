// This code by David Bellis in 2021 (inspired by Intel sample code)

#include <CL/sycl.hpp>
using namespace sycl;

#include <math.h>

event do_sqrt(queue &q, unsigned long N, unsigned long B, double *p_data_usm, buffer<double> *p_data_buffer, std::vector<event> & events) {
    
  //our nd_range
  nd_range R{range<1>{N}, range<1>{B}};
    
  if(p_data_usm)
  {
    return q.parallel_for(
      R,
      events, 
      [=](nd_item<1> it) { int i = it.get_global_id(0); p_data_usm[i] = sqrt(p_data_usm[i]); }
    );
  }
  else if(p_data_buffer)
  {
    return q.submit([&](handler& h) {
      accessor data(*p_data_buffer,h,write_only);
      h.depends_on(events);
      h.parallel_for(
        R, 
        [=](nd_item<1> it) { int i = it.get_global_id(0); data[i] = sqrt(data[i]); }
      ); 
    });
  }
  
  return event();
} 

