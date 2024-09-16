// resizeAndNormalize.h
#ifndef RESIZE_AND_NORMALIZE_H
#define RESIZE_AND_NORMALIZE_H

#include <cuda_runtime.h>
#include <npp.h>

#ifdef __cplusplus
extern "C" {
#endif
    void normalize(const Npp8u* src, float* dst, int width, int height);
#ifdef __cplusplus
}
#endif

#endif // RESIZE_AND_NORMALIZE_H
