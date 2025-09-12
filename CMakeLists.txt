cmake_minimum_required(VERSION 3.18)
project(warfeye)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

pkg_check_modules(GIO REQUIRED gio-2.0)
pkg_check_modules(LIBPORTAL REQUIRED libportal)
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)

add_subdirectory(libuiohook)
add_subdirectory(tesseract)
add_subdirectory(cpr)

add_executable(warfeye main.cpp xdp_stream.cpp)

include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

target_link_libraries(warfeye PRIVATE fmt::fmt)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Tesseract_INCLUDE_DIRS} )

target_link_libraries(warfeye PRIVATE ${GIO_LIBRARIES})
target_include_directories(warfeye PRIVATE ${GIO_INCLUDE_DIRS})

target_link_libraries(warfeye PRIVATE ${LIBPORTAL_LIBRARIES})
target_include_directories(warfeye PRIVATE ${LIBPORTAL_INCLUDE_DIRS})

target_link_libraries(warfeye PRIVATE ${PIPEWIRE_LIBRARIES})
target_include_directories(warfeye PRIVATE ${PIPEWIRE_INCLUDE_DIRS})
target_compile_options(warfeye PRIVATE ${PIPEWIRE_CFLAGS_OTHER})

target_link_libraries(warfeye PRIVATE uiohook)
target_include_directories(warfeye PRIVATE uiohook)

target_link_libraries(warfeye PRIVATE libtesseract )
target_link_libraries(warfeye PRIVATE ${TESSERACT_LIBRARIES})
target_link_libraries(warfeye PRIVATE ${TESSERACT_LINK_LIBRARIES})

target_link_libraries(warfeye PRIVATE ${OpenCV_LIBS})

target_link_libraries(warfeye PRIVATE cpr::cpr)
target_include_directories(warfeye PRIVATE cpr::cpr)
