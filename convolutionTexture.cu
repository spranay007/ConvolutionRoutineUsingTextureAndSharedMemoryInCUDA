#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "convolutionTexture_common.h"

// Define kernel parameters
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// Constant memory for the convolution kernel
__constant__ float c_Kernel[KERNEL_LENGTH_MAX];

// Function to set the convolution kernel in constant memory
extern "C" void setConvolutionKernel(float* h_Kernel, int kernelLengthUser) {
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, kernelLengthUser * sizeof(float));
}

// Kernel for row convolution
__global__ void convolutionRowsKernel(float* d_Dst, int imageW, int imageH, cudaTextureObject_t texSrc, int kernelRadiusUser) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // Allocate dynamic shared memory
    extern __shared__ float sharedMemory[];

    if (ix < imageW && iy < imageH) {
        // Compute global index
        int globalIdx = iy * imageW + ix;

        // Load data into shared memory
        sharedMemory[threadIdx.y * (blockDim.x + 2 * kernelRadiusUser) + threadIdx.x + kernelRadiusUser] = tex2D<float>(texSrc, ix + 0.5f, iy + 0.5f);

        // Load ghost elements into shared memory
        if (threadIdx.x < kernelRadiusUser) {
            sharedMemory[threadIdx.y * (blockDim.x + 2 * kernelRadiusUser) + threadIdx.x] = tex2D<float>(texSrc, ix - kernelRadiusUser + 0.5f, iy + 0.5f);
        }
        if (threadIdx.x >= blockDim.x - kernelRadiusUser) {
            sharedMemory[threadIdx.y * (blockDim.x + 2 * kernelRadiusUser) + threadIdx.x + 2 * kernelRadiusUser] = tex2D<float>(texSrc, ix + blockDim.x - kernelRadiusUser + 0.5f, iy + 0.5f);
        }

        // Synchronize threads to ensure all data is loaded into shared memory
        __syncthreads();

        float sum = 0;

        // Perform convolution using data from shared memory
        for (int k = -kernelRadiusUser; k <= kernelRadiusUser; k++) {
            sum += sharedMemory[threadIdx.y * (blockDim.x + 2 * kernelRadiusUser) + threadIdx.x + kernelRadiusUser + k] * c_Kernel[kernelRadiusUser - k];
        }

        d_Dst[globalIdx] = sum;
    }
}

// Wrapper function for row convolution GPU kernel
extern "C" void convolutionRowsGPU(float* d_Dst, cudaArray * a_Src, int imageW, int imageH, cudaTextureObject_t texSrc, int kernelRadiusUser) {
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    // Calculate shared memory size dynamically
    size_t sharedMemSize = BLOCK_SIZE_Y * (BLOCK_SIZE_X + 2 * kernelRadiusUser) * sizeof(float);

    convolutionRowsKernel << <blocks, threads, sharedMemSize >> > (d_Dst, imageW, imageH, texSrc, kernelRadiusUser);
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}

// Kernel for column convolution
__global__ void convolutionColumnsKernel(float* d_Dst, int imageW, int imageH, cudaTextureObject_t texSrc, int kernelRadiusUser) {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // Allocate dynamic shared memory
    extern __shared__ float sharedMemory[];

    if (ix < imageW && iy < imageH) {
        // Compute global index
        int globalIdx = iy * imageW + ix;

        // Load data into shared memory
        sharedMemory[(threadIdx.y + kernelRadiusUser) * blockDim.x + threadIdx.x] = tex2D<float>(texSrc, ix + 0.5f, iy + 0.5f);

        // Load ghost elements into shared memory
        if (threadIdx.y < kernelRadiusUser) {
            sharedMemory[threadIdx.y * blockDim.x + threadIdx.x] = tex2D<float>(texSrc, ix + 0.5f, iy - kernelRadiusUser + 0.5f);
        }
        if (threadIdx.y >= blockDim.y - kernelRadiusUser) {
            sharedMemory[(threadIdx.y + 2 * kernelRadiusUser) * blockDim.x + threadIdx.x] = tex2D<float>(texSrc, ix + 0.5f, iy + blockDim.y - kernelRadiusUser + 0.5f);
        }

        // Synchronize threads to ensure all data is loaded into shared memory
        __syncthreads();

        float sum = 0;

        // Perform convolution using data from shared memory
        for (int k = -kernelRadiusUser; k <= kernelRadiusUser; k++) {
            sum += sharedMemory[(threadIdx.y + kernelRadiusUser + k) * blockDim.x + threadIdx.x] * c_Kernel[kernelRadiusUser - k];
        }

        d_Dst[globalIdx] = sum;
    }
}

// Wrapper function for column convolution GPU kernel
extern "C" void convolutionColumnsGPU(float* d_Dst, cudaArray * a_Src, int imageW, int imageH, cudaTextureObject_t texSrc, int kernelRadiusUser) {
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    // Calculate shared memory size dynamically
    size_t sharedMemSize = (BLOCK_SIZE_Y + 2 * kernelRadiusUser) * BLOCK_SIZE_X * sizeof(float);

    convolutionColumnsKernel << <blocks, threads, sharedMemSize >> > (d_Dst, imageW, imageH, texSrc, kernelRadiusUser);
    getLastCudaError("convolutionColumnsKernel() execution failed\n");
}
