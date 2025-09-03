#pragma once

#include <cstdint>
#include <functional>

void init_screencast(int argc, char *argv[], std::function<void(void*,uint32_t,size_t,size_t)> callback);
