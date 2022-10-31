#ifndef SIMPLERKNN_LIBRARY_H
#define SIMPLERKNN_LIBRARY_H

#include <string>
#include <SimpleRKNN/option.h>

class simple_rknn
{
private:
    /* data */
public:
    simple_rknn();
    // load rknn with init context 
    // intput, output tenosr info
    error load_model(const std::string file);
    
    // get

    ~simple_rknn();
    
};



#endif // SIMPLERKNN_LIBRARY_H