// resizeAndNormalize.cu
#include <cuda_runtime.h>
#include "resizeAndNormalize.h"
#include "laplacian_filter.h"
#include <iostream>

#if 0
__global__ void resizeAndNormalizeKernel(const Npp8u* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dstWidth && y < dstHeight) {
        float gx = x * (srcWidth - 1) / (float)(dstWidth - 1);
        float gy = y * (srcHeight - 1) / (float)(dstHeight - 1);
        int gxi = (int)gx;
        int gyi = (int)gy;
        float c00 = src[gyi * srcWidth + gxi];
        float c10 = src[gyi * srcWidth + gxi + 1];
        float c01 = src[(gyi + 1) * srcWidth + gxi];
        float c11 = src[(gyi + 1) * srcWidth + gxi + 1];
        float c0 = c00 + (c10 - c00) * (gx - gxi);
        float c1 = c01 + (c11 - c01) * (gx - gxi);
        dst[y * dstWidth + x] = ((c0 + (c1 - c0) * (gy - gyi)) / 255.0f - 0.5f) / 0.5f;
    }
}

void resizeAndNormalize(const Npp8u* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight) {
    dim3 blockSize(16, 16);
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);
    resizeAndNormalizeKernel<<<gridSize, blockSize>>>(src, srcWidth, srcHeight, dst, dstWidth, dstHeight);
    cudaDeviceSynchronize();
}

#else
// Function to get global index in 2D configuration
__device__
int my_getGlobalIdx_2D_2D(void) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    return idy * gridDim.x * blockDim.x + idx;
}

// CUDA kernel to convert unsigned char image to float image
__global__ void convertToFloat(float *dst, const Npp8u *src, int width, int height) {
    int idx = my_getGlobalIdx_2D_2D();
    if (idx < width * height) {
        dst[idx] = static_cast<float>(src[idx]) / 255.0f;
    }
}

// Launch the kernel

void normalize(const Npp8u* src, float* dst, int width, int height, cudaStream_t resizeStream) {
    if( resizeStream == NULL ) {
        std::cerr << "resizeStream is NULL" << std::endl;
        exit(-1);
    }
    dim3 blockSize(16, 16); // 2D block size
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y); // 2D grid size
    
    convertToFloat<<<gridSize, blockSize, 0, resizeStream>>>(dst, src, width, height);


    // Check for errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "1. CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Synchronize and check for errors after kernel execution
    cudaStreamSynchronize(resizeStream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "2. CUDA error after synchronization: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "2. src " << (long)src << std::endl;
        std::cerr << "2. dst " << (long)dst << std::endl;
        std::cerr << "2. w " << width << std::endl;
        std::cerr << "2. h " << height << std::endl;
        exit(-1);
    }

}

#endif