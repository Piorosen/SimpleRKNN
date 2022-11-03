#include <spdlog/spdlog.h>

#include <SimpleRKNN/simple_rknn.h>
#include <SimpleRKNN/rknn_queue.h>
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
    
info_rknn simple_rknn::get_info() const {
    info_rknn value;
    error ret;
    // input, ouput 텐서 정보 가져옴.
    rknn_input_output_num io_num;
    ret = (error)rknn_query(this->id, (rknn_query_cmd)query::in_out_num, &io_num, sizeof(io_num));
    if (ret != error::success) { 
        spdlog::error("rknn query fail! ret={}", (int)ret);
        return value;
    }

    // 인풋 텐서 정보 가져옴.
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    value.input_batch = io_num.n_input;
    value.output_batch = io_num.n_output;

    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = (error)rknn_query(this->id, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != error::success)
        {
            spdlog::error("rknn_query fail! ret={}", (int)ret);
            return value;
        }
        value.input.push_back(*(attribute_tensor*)&input_attrs[i]);
        printRKNNTensor(&(input_attrs[i]));
    }
    value.input_tensor_size = input_attrs[0].size * io_num.n_input;
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
            return value;
        }
        value.output.push_back(*(attribute_tensor*)&output_attrs[i]);
        printRKNNTensor(&(output_attrs[i]));
    }
    value.output_tensor_size = input_attrs[0].size * io_num.n_output;

    return value;
}

simple_rknn::~simple_rknn() {
    // error ret = (error)rknn_init(&this->id, this->model, model_size, (uint32_t)priority_flag::high);
    rknn_destroy(this->id);
    spdlog::info("destory {}", id);
}

/* real inference time (us) */
uint64_t simple_rknn::get_inference_time() const { 
    error ret;
    // input, ouput 텐서 정보 가져옴.
    rknn_perf_run perf;
    ret = (error)rknn_query(this->id, (rknn_query_cmd)query::performance_run, &perf, sizeof(rknn_perf_run));
    if (ret != error::success) { 
        spdlog::error("rknn query fail! ret={}", (int)ret);
        return 0;
    }
    return perf.run_duration;
}

error simple_rknn::compute(void* tensor, tensor_format layout, tensor_type type, int convert_float,
                            std::function<void(void*, uint32_t)> callback) 
{ 
    // Set Input Data
    input i;
    i.layout = layout;
    i.type = type;
    i.convert_float = convert_float;
    i.tensor = tensor;
    i.id = this->id;
    i.batchs = this->batchs;
    i.tensor_size = this->tensor_size;

    rknn_queue::instance()->enqueue(i);
    return error::success;
}
