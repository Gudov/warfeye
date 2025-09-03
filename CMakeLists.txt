cmake_minimum_required(VERSION 3.18)
project(warfeye)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)

# Find the gio-2.0 package
pkg_check_modules(GIO REQUIRED gio-2.0)
# libportal
pkg_check_modules(LIBPORTAL REQUIRED libportal)
# pipewire
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)

add_executable(warfeye main.cpp xdp_stream.cpp)

target_link_libraries(warfeye PRIVATE ${GIO_LIBRARIES})
target_include_directories(warfeye PRIVATE ${GIO_INCLUDE_DIRS})

target_link_libraries(warfeye PRIVATE ${LIBPORTAL_LIBRARIES})
target_include_directories(warfeye PRIVATE ${LIBPORTAL_INCLUDE_DIRS})

target_link_libraries(warfeye PRIVATE ${PIPEWIRE_LIBRARIES})
target_include_directories(warfeye PRIVATE ${PIPEWIRE_INCLUDE_DIRS})
target_compile_options(warfeye PRIVATE ${PIPEWIRE_CFLAGS_OTHER})

# find_package(sdbus-c++ REQUIRED)

# target_link_libraries(warfeye PRIVATE SDBusCpp::sdbus-c++)
