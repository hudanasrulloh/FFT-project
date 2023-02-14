cmake_minimum_required(VERSION 3.10)
project(CudaMonteCarloFourierTransformIntegration)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_library(cuda_monte_carlo_fourier_transform_integration SHARED
                 src/cuda_monte_carlo_fourier_transform_integration.cu)

add_executable(test test/test.cpp)
target_link_libraries(test cuda_monte_carlo_fourier_transform_integration ${CUDA_