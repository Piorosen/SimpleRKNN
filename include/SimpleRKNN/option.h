#ifndef SIMPLERKNN_LIBRARY_ERROR_IFNO_H
#define SIMPLERKNN_LIBRARY_ERROR_IFNO_H

#include <functional>

typedef uint64_t context;



/*
    the output information for rknn_outputs_get.
*/
typedef struct _output {
    uint8_t want_float;                                 /* want transfer output data to float */
    uint8_t is_prealloc;                                /* whether buf is pre-allocated.
                                                           if TRUE, the following variables need to be set.
                                                           if FALSE, the following variables do not need to be set. */
    uint32_t index;                                     /* the output index. */
    void* buf;                                          /* the output buf for index.
                                                           when is_prealloc = FALSE and rknn_outputs_release called,
                                                           this buf pointer will be free and don't use it anymore. */
    uint32_t size;                                      /* the size of output buf. */
} output;


/*
    Definition of extended flag for rknn_init.
*/
enum class priority_flag : int { 
    /* set high priority context. */
    high = 0,
    /* set medium priority context */
    medium = 1,
    /* set low priority context. */
    low = 2,
    /* asynchronous mode.
        when enable, rknn_outputs_get will not block for too long because it directly retrieves the result of
        the previous frame which can increase the frame rate on single-threaded mode, but at the cost of
        rknn_outputs_get not retrieves the result of the current frame.
        in multi-threaded mode you do not need to turn this mode on. */
    async_mask = 4,
    /* collect performance mode.
        when enable, you can get detailed performance reports via rknn_query(ctx, 
        RKNN_QUERY_PERF_DETAIL, ...),
        but it will reduce the frame rate. */
    performance = 8,
    /* You can store the rknn model under NPU, 
        * when you call rknn_init(), you can pass the filename of model instead of model data.
        * Then you can hide your model and be invisible to the end user.
        * */
    load_model_in_npu = 16,
};

/*
    Error code returned by the RKNN API.
*/
enum class error : int { 
    success             =  0 ,
    fail                = -1 ,
    timeout             = -2 ,
    device_unavailable  = -3 ,
    malloc_fail         = -4 ,
    error_param         = -5 ,
    error_model         = -6 ,
    error_context       = -7 ,
    error_input         = -8 ,
    error_output        = -9 ,
    unmatch_device      = -10,
};

/*
    The query command for rknn_query
*/
enum class query : int {
    in_out_num          = 0, /* query the number of input & output tensor. */
    input_tensor        = 1,/* query the attribute of input tensor. */
    output_tensor       = 2,/* query the attribute of output tensor. */
    performance_detail  = 3, /* query the detail performance, need set
                                                           RKNN_FLAG_COLLECT_PERF_MASK when call rknn_init,
                                                           this query needs to be valid after rknn_outputs_get. */
    performance_run     = 4, /* query the time of run,
                                                           this query needs to be valid after rknn_outputs_get. */
    sdk_version         = 5, /* query the sdk & driver version */
    // latest_query = 6
};

enum class tensor_type : int {
    float32 = 0,                            /* data type is float32. */
    float16,                                /* data type is float16. */
    int8,                                   /* data type is int8. */
    uint8,                                  /* data type is uint8. */
    int16,                                  /* data type is int16. */
};

/*
    the quantitative type.
*/
enum class tensor_quantize_type : int  {
    none = 0,                           /* none. */
    dynamic_fixed_point,                /* dynamic fixed point. */
    affine_asymmetric,                  /* asymmetric affine. */
};

/*
    the tensor data format.
*/
enum class tensor_format : int  {
    nchw = 0,                               /* data format is NCHW. */
    nhwc,                                   /* data format is NHWC. */
};

typedef struct _input { 
    context id;
    void* tensor;
    uint32_t tensor_size;
    tensor_type type;
    tensor_format layout;
    uint8_t convert_float;
    int batchs;

    std::function<void(void*, uint32_t)> callback;

} input;

struct attribute_tensor {
    uint32_t index;                                     /* input parameter, the index of input/output tensor,
                                                           need set before call rknn_query. */

    uint32_t n_dims;                                    /* the number of dimensions. */
    uint32_t dims[16];                       /* the dimensions array. */
    char name[256];                       /* the name of tensor. */

    uint32_t n_elems;                                   /* the number of elements. */
    uint32_t size;                                      /* the bytes size of tensor. */

    tensor_format fmt;                             /* the data format of tensor. */
    tensor_type type;                              /* the data type of tensor. */
    tensor_quantize_type qnt_type;                      /* the quantitative type of tensor. */
    int8_t fl;                                          /* fractional length for RKNN_TENSOR_QNT_DFP. */
    uint32_t zp;                                        /* zero point for RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC. */
    float scale;                                        /* scale for RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC. */
};

#endif // SIMPLERKNN_LIBRARY_ERROR_IFNO_H