#ifndef SIMPLERKNN_LIBRARY_RKNN_QUEUE_H
#define SIMPLERKNN_LIBRARY_RKNN_QUEUE_H

// 동기적으로 순차적으로 처리 할 수 있는 시스템을 구축합니다.
// 싱글톤으로 구현해야하며, 뭐.. 등등? 해야겠죠?
#include <thread>

#include <SimpleRKNN/simple_rknn.h>
#include <msd/channel.hpp>
namespace rknn { 
class rknn_queue { 
private:
    std::thread loop;
#ifdef RKNN_DEVICE_BUFFER
    msd::channel<input> chan{8}; // buffered
#else
    msd::channel<input> chan{0}; // unbuffered
#endif
    bool stop_signal = false;

    rknn_queue(const rknn_queue& oth) = delete;
    rknn_queue() {}

public:
    static rknn_queue* instance() { 
        static rknn_queue inst;
        return &inst;
    }

    void enqueue(input data);
    
    void run_loop();
    void close_loop();
};
}

#endif // SIMPLERKNN_LIBRARY_RKNN_QUEUE_H
