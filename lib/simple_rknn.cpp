#include <spdlog/spdlog.h>
#include <functional>

#include <SimpleRKNN/simple_rknn.h>
#include <rknn/rknn_api.h>


unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}


void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}



    // load rknn with init context 
    // intput, output tenosr info
error simple_rknn::load_model(const std::string file) { 
    // model 정보 로드
    this->model = ::load_model(file.c_str(), &this->model_size);
    error ret = (error)rknn_init(&this->id, this->model, model_size, (uint32_t)priority_flag::high);
    if (ret != error::success) { 
        spdlog::error("rknn init fail! ret={}", (int)ret);
        return ret;
    }

    // input, ouput 텐서 정보 가져옴.
    rknn_input_output_num io_num;
    ret = (error)rknn_query(this->id, (rknn_query_cmd)query::in_out_num, &io_num, sizeof(io_num));
    if (ret != error::success) { 
        spdlog::error("rknn query fail! ret={}", (int)ret);
        return ret;
    }

    // 인풋 텐서 정보 가져옴.
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    this->batchs =  io_num.n_input;

    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = (error)rknn_query(this->id, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != error::success)
        {
            spdlog::error("rknn_query fail! ret={}", (int)ret);
            return ret;
        }
        printRKNNTensor(&(input_attrs[i]));
        this->tensor_size = input_attrs[0].size;
    }

    // 출력 텐서 정보 가져옴.
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = (error)rknn_query(this->id, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != error::success)
        {
            spdlog::error("rknn_query fail! ret={}", (int)ret);
            return ret;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

    return error::success;
}
    
error simple_rknn::compute(void* tensor, 
                               uint32_t tensor_size, 
                               tensor_type type, 
                               tensor_format layout,
                               uint8_t convert_float,
                               std::function<void(void*, uint32_t)> callback) 
{ 
    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = (rknn_tensor_type)type;
    inputs[0].size = this->tensor_size;
    inputs[0].fmt = (rknn_tensor_format)layout;
    inputs[0].buf = tensor;
    error ret = (error)rknn_inputs_set(this->id, this->batchs, inputs);
    if (ret != error::success)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return ret;
    }

    // Run
    // printf("rknn_run\n");
    ret = (error)rknn_run(this->id, nullptr);
    if (ret != error::success)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return ret;
    }

    // Get Output
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = (error)rknn_outputs_get(this->id, 1, outputs, NULL);
    if (ret != error::success)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return ret;
    }

    callback(outputs[0].buf, outputs[0].size);
    rknn_outputs_release(this->id, 1, outputs);
    return error::success;
}
