#ifndef __NUFFT_H__
#define __NUFFT_H__

#include "utils.h"

/**
 * Computes the non-uniform discrete Fourier transform of the given input data at the
 * specified points using the Type 3 Nonuniform FFT algorithm.
 *
 * @param[in] num_points The number of input data points.
 * @param[in] num_dims The number of dimensions of the input data.
 * @param[in] points The array of input data points.
 * @param[in] values The array of input data values.
 * @param[in] num_samples The number of samples to use in the output FFT.
 * @param[in] tol The tolerance of the algorithm.
 * @param[in] batch_size The number of points to compute per batch.
 * @param[out] output The array to store the computed FFT values.
 * @param[in] use_double_precision Whether to use double precision for the computation.
 * @param[in] device_id The index of the CUDA device to use for the computation.
 *
 * @return The number of elements written to the output array.
 */
int compute_nufft(int num_points, int num_dims, const double *points, const double *values,
                  int num_samples, double tol, int batch_size, double *output,
                  bool use_double_precision, int device_id);

/**
 * Integrates the given input function using Monte Carlo integration.
 *
 * @param[in] integrand The integrand to evaluate.
 * @param[in] integrand_params The parameters for the integrand.
 * @param[in] num_dims The number of dimensions of the integrand.
 * @param[in] num_samples The number of samples to use in the Monte Carlo integration.
 * @param[in] lower_bounds The lower bounds of the integration domain.
 * @param[in] upper_bounds The upper bounds of the integration domain.
 * @param[out] result The array to store the computed integral result.
 * @param[in] device_id The index of the CUDA device to use for the computation.
 *
 * @return True if the computation succeeded, false otherwise.
 */
bool integrate(const integrand_t integrand, const void *integrand_params, int num_dims,
               int num_samples, const double *lower_bounds, const double *upper_bounds,
               double *result, int device_id);

#endif // __NUFFT_H__

