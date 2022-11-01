#include <spdlog/spdlog.h>
#include <SimpleRKNN/rknn_queue.h>
#include <rknn/rknn_api.h>

void rknn_queue::enqueue(input data)
{ 
    data >> this->chan;
}
    
void rknn_queue::run_loop() { 
    stop_signal = false;
    loop = std::thread([this]() { 
        for (const auto& out : this->chan) { 
            if (stop_signal == true) { 
                return;
            }
            rknn_input inputs[1];
            memset(inputs, 0, sizeof(inputs));
            inputs[0].index = 0;
            inputs[0].type = (rknn_tensor_type)out.type;
            inputs[0].size = out.tensor_size;
            inputs[0].fmt = (rknn_tensor_format)out.layout;
            inputs[0].buf = out.tensor;
            error ret = (error)rknn_inputs_set(out.id, out.batchs, inputs);
            if (ret != error::success)
            {
                spdlog::error("rknn input fail! ret={}", (int)ret);
                // return ret;
            }

            // Run
            // printf("rknn_run\n");
            ret = (error)rknn_run(out.id, nullptr);
            if (ret != error::success)
            {
                spdlog::error("rknn run fail! ret={}", (int)ret);
                // return ret;
            }

            // Get Output
            rknn_output outputs[1];
            memset(outputs, 0, sizeof(outputs));
            outputs[0].want_float = 1;
            ret = (error)rknn_outputs_get(out.id, 1, outputs, NULL);
            if (ret != error::success)
            {
                spdlog::error("rknn output get fail! ret={}", (int)ret);
                // return ret;
            }
            out.callback(outputs[0].buf, outputs[0].size);
            rknn_outputs_release(out.id, 1, outputs);
            
            if (stop_signal == true) { 
                return;
            }
        }
    });
}
void rknn_queue::close_loop() {
    stop_signal = true;
}