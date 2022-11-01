# Simple RKNN

# How to build

## Requirements

> 1. CMake 3.4 ~<br>
> 2. make<br>
> 3. RK3399Pro or Asus Tinker Edge R<br>

## Build

> $ git clone https://github.com/Piorosen/SimpleRKNN <br>
> $ cd SimpleRKNN && mkdir build && cd build<br>
> $ cmake .. && make -j8 && make install<br>

# Example

```cpp
#include <iostream>
#include <SimpleRKNN/simple_rknn.h>

int main() { 
    simple_rknn lib;
    if (lib.load_model("mobile_net.rknn") != error::success) { 
        std::cout << "error! \n";
        return 0;        
    }

    lib.compute(tensor_data);
}

```
