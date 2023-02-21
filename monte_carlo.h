#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <curand_kernel.h>

typedef double (*Integrand)(double*);

class MonteCarlo {
  private:
    int num_dims_;
    double* a_;
    double* b_;
    curandState* rand_states_;
    double integral_;
    int num_points_;
    double* device_points_;
    double* device_values_;
    double* host_values_;

  public:
    MonteCarlo(int num_dims, double* a, double* b, int num_points);

    void setIntegrand(Integrand integrand);

    void integrate();

    double getIntegral();

    ~MonteCarlo();
};

#endif // MONTE_CARLO_H

