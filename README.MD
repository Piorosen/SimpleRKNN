# Simple RKNN

# How to build

## Requirements

> 1. CMake 3.4 ~<br>
> 2. make<br>
> 3. C++ 11<br>
> 4. RK3399Pro or Asus Tinker Edge R<br>

## Build

> $ git clone https://github.com/Piorosen/SimpleRKNN <br>
> $ cd SimpleRKNN && mkdir build && cd build<br>
> $ cmake .. && make -j8 && make install<br>

# Example

```cpp
#include <iostream>
#include <SimpleRKNN/simple_rknn.h>

int main() { 
    // async compute loop
    rknn::run_loop();
    
    // inference query class
    rknn::simple_rknn lib;
    
    // load model on inference engine 
    // only read rknn files
    lib.load_model("mobilenet_v1.rknn");
    // load image for rknn by stbimage
    auto image = lib.load_image("dog_224x224.jpg", rknn::tensor_format::nchw);
    
    // 30 times, test inference and measure time
    for (int i = 0; i < 30; i++) { 
        auto compute = high_resolution_clock::now();
        // flag are converted from async to sync.
        bool flag = false;
        lib.compute(image, rknn::tensor_format::nchw, rknn::tensor_type::uint8, 0, [&lib, &flag, &compute](auto buf, auto size) {
            cout << (high_resolution_clock::now() - compute).count() << "\n";
            flag = true;
        });
        // wait. When the flag is changed to true.
        while(!flag);
    }
    
    lib.free_image(image);
    
    rknn::close_loop();
    
    return 0;
}
```
