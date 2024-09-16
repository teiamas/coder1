#include <cuda_runtime.h>
#include <iostream>
#include "gpu_frame_diff.h"

// CUDA kernel for frame difference calculation
__global__ void frame_diff_kernel(const unsigned char* frame1, const unsigned char* frame2, unsigned int* diff, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        atomicAdd(diff, abs(frame1[idx] - frame2[idx]));
    }
}

class GPUHandler {
public:
    cudaStream_t stream;
    unsigned char *d_frame1, *d_frame2;
    int size;

    void engage(int s) {
        size=s;
        cudaStreamCreate(&stream);
        cudaMalloc(&d_frame1, size * sizeof(unsigned char));
        cudaMalloc(&d_frame2, size * sizeof(unsigned char));
    }

    void disengage() {
        cudaFree(d_frame1);
        cudaFree(d_frame2);
        cudaStreamDestroy(stream);
    }
};

GPUHandler gpuHandler;

extern "C" void engage_frame_diff(int size) {
    gpuHandler.engage(size);
}

extern "C" void disengage_frame_diff() {
    gpuHandler.disengage();
}

extern "C" unsigned int do_frame_diff(const unsigned char* frame1, const unsigned char* frame2) {
    unsigned int h_diff = 0;
    unsigned int* d_diff;

    // Allocate memory for the difference on the GPU
    cudaMalloc(&d_diff, sizeof(unsigned int));
    cudaMemset(d_diff, 0, sizeof(unsigned int));

    // Copy frames to GPU
    cudaMemcpyAsync(gpuHandler.d_frame1, frame1, gpuHandler.size * sizeof(unsigned char), cudaMemcpyHostToDevice, gpuHandler.stream);
    cudaMemcpyAsync(gpuHandler.d_frame2, frame2, gpuHandler.size * sizeof(unsigned char), cudaMemcpyHostToDevice, gpuHandler.stream);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (gpuHandler.size + blockSize - 1) / blockSize;

    // Launch kernel
    frame_diff_kernel<<<gridSize, blockSize, 0, gpuHandler.stream>>>(gpuHandler.d_frame1, gpuHandler.d_frame2, d_diff, gpuHandler.size);

    // Copy result back to host
    cudaMemcpyAsync(&h_diff, d_diff, sizeof(unsigned int), cudaMemcpyDeviceToHost, gpuHandler.stream);

    // Synchronize stream
    cudaStreamSynchronize(gpuHandler.stream);

    // Free GPU memory
    cudaFree(d_diff);

    return h_diff;
}