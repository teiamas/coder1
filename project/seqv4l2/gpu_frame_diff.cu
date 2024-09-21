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
    unsigned char *d_frame1, *d_frame2; // Device pointers to the frames
    unsigned int *d_diff;               // Device pointer to the difference
    int size;                           // Size of the frame
    unsigned int h_diff = 0;            // Host variable to store the difference

    void engage(int s) {
        size=s;
        cudaStreamCreate(&stream);
        cudaMalloc(&d_frame1, size * sizeof(unsigned char));
        cudaMalloc(&d_frame2, size * sizeof(unsigned char));
        cudaMalloc(&d_diff, sizeof(unsigned int)); // Allocate memory for the difference
        cudaHostAlloc((void**)&h_diff, sizeof(unsigned int), cudaHostAllocDefault);
    }

    void disengage() {
        cudaFree(d_frame1);
        cudaFree(d_frame2);
        cudaFree(d_diff); // Free memory for the difference
        cudaStreamDestroy(stream);
        cudaFreeHost(&h_diff);
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
    // Reset the difference on the GPU
    cudaMemsetAsync(gpuHandler.d_diff, 0, sizeof(unsigned int), gpuHandler.stream);

    // Copy frames to GPU
    cudaMemcpyAsync(gpuHandler.d_frame1, frame1, gpuHandler.size * sizeof(unsigned char), cudaMemcpyHostToDevice, gpuHandler.stream);
    cudaMemcpyAsync(gpuHandler.d_frame2, frame2, gpuHandler.size * sizeof(unsigned char), cudaMemcpyHostToDevice, gpuHandler.stream);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (gpuHandler.size + blockSize - 1) / blockSize;

    // Launch kernel
    frame_diff_kernel<<<gridSize, blockSize, 0, gpuHandler.stream>>>(gpuHandler.d_frame1, gpuHandler.d_frame2, gpuHandler.d_diff, gpuHandler.size);

    // Copy result back to host
    cudaMemcpyAsync(&gpuHandler.h_diff, gpuHandler.d_diff, sizeof(unsigned int), cudaMemcpyDeviceToHost, gpuHandler.stream);

    // Synchronize stream
    cudaStreamSynchronize(gpuHandler.stream);

    // return the two frames difference
    return gpuHandler.h_diff;;
}