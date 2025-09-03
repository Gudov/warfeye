#include "xdp_stream.hpp"

#include <iostream>

int main(int argc, char *argv[]) {
    init_screencast(argc, argv, [](void*data,uint32_t size,size_t w,size_t h) {
        std::cout << "got frame " << size << std::endl;
    });
    return 0;
}