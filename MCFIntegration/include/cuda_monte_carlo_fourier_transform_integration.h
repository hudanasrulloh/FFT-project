//cuda_monte_carlo_fourier_transform_integration.h 

#ifndef CUDA_MONTE_CARLO_FOURIER_TRANSFORM_INTEGRATION_H
#define CUDA_MONTE_CARLO_FOURIER_TRANSFORM_INTEGRATION_H

#include <cuda_runtime.h>

namespace CudaMonteCarloFourierTransformIntegration {

// Monte Carlo integration
__device__ __host__
void monte_carlo(int num_samples, int num_dimensions, int size, float *input, float *output);

// Fourier transform
__device__ __host__
void fourier_transform(int size, float *input, float *output);

} // namespace CudaMonteCarloFourierTransformIntegration

#endif // CUDA_MONTE_CARLO_FOURIER_TRANSFORM_INTEGRATION_H