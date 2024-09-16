#ifndef GPU_FRAME_DIFF_H
#define GPU_FRAME_DIFF_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// External C functions to interface with the GPUHandler class
void engage_frame_diff(int size);
void disengage_frame_diff();
unsigned int do_frame_diff(const unsigned char* frame1, const unsigned char* frame2);

#ifdef __cplusplus
}
#endif

#endif // SEARCH_MAX_FRAME_DIFF_H