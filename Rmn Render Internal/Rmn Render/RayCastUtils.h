#ifndef RAY_CAST_UTILS_H_
#define RAY_CAST_UTILS_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"

#include <stdexcept>
#include <string>


__device__ int pointNormal(dim3 dataSize, float3 point)
{
    int i = point.x, j = point.y, k = point.z;

    i = i < 0 ? 0 : (i >= dataSize.x ? dataSize.x - 1 : i);
    j = j < 0 ? 0 : (j >= dataSize.y ? dataSize.y - 1 : j);
    k = k < 0 ? 0 : (k >= dataSize.z ? dataSize.z - 1 : k);

    return i * dataSize.y * dataSize.z + j * dataSize.z + k;
}

__device__ float composeRGBA(float prev, float color, float alpha)
{
    return prev * alpha + color * (1 - alpha);
}

__device__ unsigned char colorFloatToByte(float color)
{
    float f = 255 * color;

    if (f < 0) f = 0.0f;
    if (f > 255) f = 255.0f;

    return floor(f);
}

__device__ float length(float x, float y, float z)
{
    return sqrt(x * x + y * y + z * z);
}

__device__ float vectorLength(float3 vector)
{
    return sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

__device__ float3 normalizeVector(float3 vector)
{
    float3 normalizedVector;
    float length = vectorLength(vector);

    normalizedVector.x = vector.x / length;
    normalizedVector.y = vector.y / length;
    normalizedVector.z = vector.z / length;

    return normalizedVector;
}

__device__ float imageCoordonateToViewPlane(size_t imageCoordonate, size_t imageSize, float viewPlaneSize)
{
    return (float)imageCoordonate * viewPlaneSize / (float)imageSize - viewPlaneSize / 2.0f;
}

#endif // !RAY_CAST_UTILS_H_

