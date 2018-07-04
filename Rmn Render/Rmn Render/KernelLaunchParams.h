#ifndef KERNEL_LAUNCH_PARAMS_H_
#define KERNEL_LAUNCH_PARAMS_H_

#include "cuda.h"
#include "cuda_runtime.h"

typedef struct _KernelLaunchParams
{
    unsigned char* dev_rmnData;
    float3* dev_normals;
    uint2 imageDim;
    dim3 rmnDim;
    int2 rotation;
    float ratio;
}KernelLaunchParams;

#endif // !KERNEL_LAUNCH_PARAMS_H_
