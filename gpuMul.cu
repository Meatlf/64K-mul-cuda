/*
 ============================================================================
 Name        : gpuMul.cu
 Author      : ttz 
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "NTT.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <stdint.h>
#include <stdlib.h>
#include "ModP.h"
#include "kernel.h"
#include "Base.h"
#include <NTL/ZZ.h>
#include <NTL/ZZX.h>
#include <ctime>
using namespace NTL;
using namespace cuHE;
		
#define len 256
#define len_64K 65536
#define len_64K_half 32768
int main(){
	clock_t t1,t2;
//	t1=clock();
	const ZZ P=to_ZZ(0xffffffff00000001);
	const ZZ root_256=to_ZZ((uint64)14041890976876060974);
	const ZZ root_64K=to_ZZ((uint64)15893793146607301539);

	uint32 *hx_l,*dx_l;
	uint64 *ht_l,*dt_l;
	uint64 *hy_l,*dy_l;

	uint32 *hx_r,*dx_r;
	uint64 *ht_r,*dt_r;
	uint64 *hy_r,*dy_r;
  
	uint64 *hx,*dx;
	uint64 *ht,*dt;
	uint64 *hy,*dy;

	uint64 *h_roots,*d_roots;
	uint64 *h_roots_64K,*d_roots_64K;
  uint64 *h_roots_64K_inverse,*d_roots_64K_inverse;  
	
	dim3 BlockDim(16,16);

  cudaStream_t stream[2];
	for(int i=0;i<2;i++)
			cudaStreamCreate(&stream[i]);
	t1=clock();	
	cudaMallocHost(&hx_l,len_64K*sizeof(uint32));
	cudaMallocHost(&ht_l,len_64K*sizeof(uint64));
	cudaMallocHost(&hy_l,len_64K*sizeof(uint64));
	
  cudaMallocHost(&hx_r,len_64K*sizeof(uint32));
  cudaMallocHost(&ht_r,len_64K*sizeof(uint64));
  cudaMallocHost(&hy_r,len_64K*sizeof(uint64));	

	cudaMallocHost(&hx,len_64K*sizeof(uint64));
	cudaMallocHost(&ht,len_64K*sizeof(uint64));
	cudaMallocHost(&hy,len_64K*sizeof(uint64));

	cudaMallocHost(&h_roots,len*sizeof(uint64));
  cudaMallocHost(&h_roots_64K,len_64K*sizeof(uint64));
	cudaMallocHost(&h_roots_64K_inverse,len_64K*sizeof(uint64));
	t2=clock();
//	t1=clock();
	cudaMalloc(&dx_l,len_64K*sizeof(uint32));
	cudaMalloc(&dt_l,len_64K*sizeof(uint64));
	cudaMalloc(&dy_l,len_64K*sizeof(uint64));

	 cudaMalloc(&dx_r,len_64K*sizeof(uint32));
   cudaMalloc(&dt_r,len_64K*sizeof(uint64));
   cudaMalloc(&dy_r,len_64K*sizeof(uint64));
	 
	 cudaMalloc(&dx,len_64K*sizeof(uint64));
	 cudaMalloc(&dt,len_64K*sizeof(uint64));
	 cudaMalloc(&dy,len_64K*sizeof(uint64));

 	 cudaMalloc(&d_roots,len*sizeof(uint64));
	 cudaMalloc(&d_roots_64K,len_64K*sizeof(uint64));
	 cudaMalloc(&d_roots_64K_inverse,len_64K*sizeof(uint64));	
//		t2=clock();
  for(int i=0;i<len_64K_half;i++){
		hx_l[i]=2;
	 	ht_l[i]=0;
		hy_l[i]=0;
		
		hx_r[i]=2;
		ht_r[i]=0;
		hy_r[i]=0;
		
	  hx[i]=0;
		ht[i]=0;
		hy[i]=0;
	}

  for(int i=0;i<16;i++){
		for(int k=0;k<16;k++){
				conv(h_roots[16*i+k],PowerMod(root_256,i*k,P));
		//test:		cout<<h_roots[16*i+k]<<endl;
		}
	}

	for(int i=0;i<256;i++){
		for(int k=0;k<256;k++){
				conv(h_roots_64K[256*i+k],PowerMod(root_64K,i*k,P));
		}
	}

	for(int i=0;i<256;i++){
		for(int k=0;k<256;k++){
				conv(h_roots_64K_inverse[256*i+k],PowerMod(root_64K,65535*i*k,P));
		}
	}

  	cudaMemcpyAsync(dx_l,hx_l,len_64K*sizeof(uint32),cudaMemcpyHostToDevice,stream[0]);
		cudaMemcpyAsync(dx_r,hx_r,len_64K*sizeof(uint32),cudaMemcpyHostToDevice,stream[1]);

  	cudaMemcpyAsync(d_roots,h_roots,len*sizeof(uint64),cudaMemcpyHostToDevice,stream[0]);
		cudaMemcpyAsync(d_roots,h_roots,len*sizeof(uint64),cudaMemcpyHostToDevice,stream[1]);
		cudaMemcpyAsync(d_roots_64K,h_roots_64K,len_64K*sizeof(uint64),cudaMemcpyHostToDevice,stream[0]);
		cudaMemcpyAsync(d_roots_64K,h_roots_64K,len_64K*sizeof(uint64),cudaMemcpyHostToDevice,stream[1]);
		//t1=clock();	
		NTT_Kernel_64K_points_1_one<<<16,BlockDim,0,stream[0]>>>(dt_l,dx_l,d_roots);
		NTT_Kernel_64K_points_2_one<<<16,BlockDim,0,stream[0]>>>(dy_l,dt_l,d_roots,d_roots_64K);
		NTT_Kernel_64K_points_1_second<<<16,BlockDim,0,stream[1]>>>(dt_r,dx_r,d_roots);
		NTT_Kernel_64K_points_2_second<<<16,BlockDim,0,stream[1]>>>(dy_r,dt_r,d_roots,d_roots_64K);
  	cudaMemcpyAsync(hy_l,dy_l,len_64K*sizeof(uint64),cudaMemcpyDeviceToHost,stream[0]);
		cudaMemcpyAsync(hy_r,dy_r,len_64K*sizeof(uint64),cudaMemcpyDeviceToHost,stream[1]);	

		dotMul(hx,hy_l,hy_r);
		cudaMemcpy(d_roots_64K_inverse,h_roots_64K_inverse,len_64K*sizeof(uint64),cudaMemcpyHostToDevice);
		cudaMemcpy(dx,hx,len_64K*sizeof(uint64),cudaMemcpyHostToDevice);
		intt_64K(dy,dt,dx,d_roots,d_roots_64K_inverse);
		cudaMemcpy(hy,dy,len_64K*sizeof(uint64),cudaMemcpyDeviceToHost);
//	t2=clock();
//	cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<endl;
		uint64_t carry=0;
		t1=clock();
		for(int i=0;i<65536;i++){
			hy[i]+=carry;
			uint64_t current=hy[i];
			carry=current>>32;
			hy[i]=hy[i]&0xFFFFFFFF;
 // 		cout<<hy[i]<<endl;
	  }
		t2=clock();
		cout<<(double)(t2-t1)/CLOCKS_PER_SEC<<endl;

	cudaFree(dx_l);
	cudaFree(dt_l);
	cudaFree(dy_l);

	cudaFree(dx_r);
	cudaFree(dt_r);
	cudaFree(dy_r);
	
  cudaFree(dx);
	cudaFree(dt);	
	cudaFree(dy);

	cudaFree(d_roots);
	cudaFree(d_roots_64K);
	cudaFree(d_roots_64K_inverse);
	
	cudaFreeHost(hx_l);
	cudaFreeHost(hx_l);
	cudaFreeHost(hx_l);

	cudaFreeHost(hx_r);
	cudaFreeHost(ht_r);
	cudaFreeHost(hy_r);
	
	cudaFreeHost(hx);
	cudaFreeHost(ht);
	cudaFreeHost(hy);

	cudaFreeHost(h_roots);
	cudaFreeHost(h_roots_64K);
	cudaFreeHost(h_roots_64K_inverse);
	return 0;
}
