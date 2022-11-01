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
    
    error compute(input data);    

};

#endif // SIMPLERKNN_LIBRARY_H