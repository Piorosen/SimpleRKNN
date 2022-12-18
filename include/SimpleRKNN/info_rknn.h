#ifndef SIMPLERKNN_LIBRARY_IFNO_RKNN_H
#define SIMPLERKNN_LIBRARY_IFNO_RKNN_H

#include <vector>
#include <SimpleRKNN/option.h>
namespace rknn { 

struct info_rknn
{
    uint32_t input_batch;                                   /* the number of input. */
    uint32_t input_tensor_size;

    uint32_t output_batch;                                  /* the number of output. */
    uint32_t output_tensor_size;
    std::vector<attribute_tensor> input;
    std::vector<attribute_tensor> output;
};
}

#endif // SIMPLERKNN_LIBRARY_IFNO_RKNN_H
