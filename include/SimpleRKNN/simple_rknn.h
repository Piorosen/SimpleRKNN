#ifndef SIMPLERKNN_LIBRARY_H
#define SIMPLERKNN_LIBRARY_H

#include <string>
#include <tuple>

#include <SimpleRKNN/option.h>  
#include <SimpleRKNN/info_rknn.h>


namespace rknn { 

void run_loop();
void close_loop();


class simple_rknn
{
private:
    context id;
    unsigned char* model;
    int model_size;

    int batchs, tensor_size;
    info_rknn info;

public:
    simple_rknn() { info.input_tensor_size = 0; }
    ~simple_rknn();
    
    // load rknn with init context 
    // intput, output tenosr info
    error load_model(const std::string file);
    
    void* load_image(const char *image_path, tensor_format layout);
    void free_image(void* image);

    info_rknn get_info() const;

    /* real inference time (us) */
    uint64_t get_inference_time() const;
    
    error compute(void* tensor, tensor_format layout, tensor_type type, int convert_float = 0,
                            std::function<void(void*, uint32_t)> callback = nullptr);

};
}
#endif // SIMPLERKNN_LIBRARY_H
