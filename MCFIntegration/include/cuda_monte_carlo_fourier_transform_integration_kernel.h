#ifndef CUDA_MONTE_CARLO_FOURIER_TRANSFORM_INTEGRATION_KERNEL_H
#define CUDA_MONTE_CARLO_FOURIER_TRANSFORM_INTEGRATION_KERNEL_H

namespace CudaMonteCarloFourierTransformIntegration {

__global__
void monte_carlo_kernel(int num_samples, int num_dimensions, int size, float *input, float *output);

__global__
void fourier_transform_kernel(int size, float *input, float *output);

} // namespace CudaMonteCarloFourierTransformIntegration

#endif // CUDA_MONTE_CARLO_FOURIER_TRANSFORM_INTEGRATION_KERNEL_H