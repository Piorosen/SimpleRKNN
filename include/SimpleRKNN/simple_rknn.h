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

    // info_rknn get_info() const;
    
    error compute(void* tensor, tensor_format layout, tensor_type type, int convert_float = 0,
                            std::function<void(void*, uint32_t)> callback = nullptr);

};

#endif // SIMPLERKNN_LIBRARY_H