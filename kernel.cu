#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "ModP.h"
#include "kernel.h"

using namespace cuHE;
__global__ void dotMul_kernel (uint64_t* z, uint64_t* x, uint64_t* y ){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

		z[index]=x[index]*y[index];
	
}


void dotMul(uint64_t* z, uint64_t* x, uint64_t* y){
	dotMul_kernel<<<256,256>>>(z,x, y);
}
