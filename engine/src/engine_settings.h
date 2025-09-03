#ifndef RT_SETTINGS_H
#define RT_SETTINGS_H

#include <iostream>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>

const double INF = std::numeric_limits<double>::infinity();
const double PI = 3.1415926535897932385;

__host__ __device__ inline double degrees_to_radians(double degrees) { return degrees * PI / 180.0; }
__host__ __device__ inline double random_double() { return std::rand() / (RAND_MAX + 1.0); }
__host__ __device__ inline double random_double(double min, double max) { return min + (max-min)*random_double(); }

#include "vec3.h"
#endif