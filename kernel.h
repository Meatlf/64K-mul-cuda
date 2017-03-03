#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "ModP.h"

__global__ void dotMul_kernel (uint64_t* z, uint64_t* x, uint64_t* y);

void dotMul(uint64_t* z, uint64_t* x, uint64_t* y);
