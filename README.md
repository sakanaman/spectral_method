# Spectral Method 
The example for solving 2D Vorticity equation with spectral method

![vortex](https://user-images.githubusercontent.com/42662735/224125455-f9baf235-52fc-43d1-9b73-3b4de6e96629.gif)

## Feature

- Using FFTW3 (SIMD, Multi-Threads) 
- Examples for spectral method

## Requirements

- CMake
- Linux or MacOS
- C++ compiler (gcc, clang,..etc)

## Notation for usage 
Add fftw options after you make build directory for you to enable SIMD and multi-threads.

```
cmake .. -DENABLE_AVX2=ON -DENABLE_THREADS=ON
```
