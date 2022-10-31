#ifndef SIMPLERKNN_LIBRARY_H
#define SIMPLERKNN_LIBRARY_H

#include <string>
#include <tuple>

#include <SimpleRKNN/option.h>

class simple_rknn
{
private:
    context id;
    unsigned char* model;
    int model_size;

    int batchs, tensor_size;
public:
    // load rknn with init context 
    // intput, output tenosr info
    error load_model(const std::string file);
    
    error compute(void* tensor, 
                               uint32_t tensor_size, 
                               tensor_type type = tensor_type::uint8, 
                               tensor_format layout = tensor_format::nhwc,
                               uint8_t convert_float = 0,
                               std::function<void(void*, uint32_t)> callback = nullptr);

    error release(output* output);
    
};



#endif // SIMPLERKNN_LIBRARY_H