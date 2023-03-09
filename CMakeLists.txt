cmake_minimum_required(VERSION 3.16.3)

project(spectral_partial)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-O3")

configure_file(
    "${CMAKE_SOURCE_DIR}/include/path.h.in"
    "${CMAKE_SOURCE_DIR}/include/path.hpp"
)

# fftw
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/fftw-3.3.10)

# main 
add_executable(main main.cpp)
target_include_directories(main PUBLIC include)

# lib
add_library(fft_vte src/spectral.cpp)
target_include_directories(fft_vte PUBLIC ext/fftw-3.3.10/api)
target_include_directories(fft_vte PUBLIC include)
target_link_libraries(fft_vte fftw3 fftw3_threads m)


target_link_libraries(main fft_vte)