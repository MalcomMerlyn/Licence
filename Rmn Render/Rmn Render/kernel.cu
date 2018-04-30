#include <GL/glut.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"

#include "GpuGlInteropAnim.h"
#include "RmnDatasetFileLoader.h"
#include "FpsDisplay.h"

#include <stdio.h>

#include <exception>
#include <iostream>
#include <vector>

#define R 0
#define G 1
#define B 2
#define A 3

using std::cout;
using std::cerr;
using std::endl;
using std::getline;
using std::exception;
using std::string;
using std::runtime_error;
using std::vector;

dim3 rmnDim;

unsigned int imageHeigth = 512;
unsigned int imageWidth = 512;

const int dim = 512;

FpsDisplay fpsDisplay({imageHeigth, imageWidth});

__device__ float4 colors[10];
__device__ uint2 colormap[10];
__device__ size_t colormapLength = 0;

__device__ unsigned int getColormapValue(size_t key)
{
    return colormap[key].x;
}

__device__ unsigned int getColormapColor(size_t key)
{
    return colormap[key].y;
}

__global__ void setColormapValue(size_t key, unsigned int value)
{
    colormap[key].x = value;
    colormapLength++;
}

__global__ void setColormapColor(size_t key, unsigned int color)
{
    colormap[key].y = color;
}

__device__ float getColorValue(size_t key, size_t color)
{
    return ((float*)(colors + key))[color];
}

__global__ void setColorValue(size_t key, size_t color, float value)
{
    ((float*)(colors + key))[color] = value;
}

__device__ int getPositionForColormapEntryValue(float value)
{
    for (size_t i = 0; i < colormapLength; i++)
    {
        if (getColormapValue(i) > value)
            return i - 1;
    }

    return -1;
}

__device__ const float pi = 3.141592653f;
__device__ const float epsilon = 0.001f;
__device__ const float step = 0.1f;
__device__ const float ambient = 0.3f;
__device__ const float diffuse = 0.7f;

__global__ void simpleCudaGreenRipple(uchar4* ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x / (float)dim - 0.5f;
    float fy = y / (float)dim - 0.5f;
    unsigned char red = 128 + 127 * sin(abs(100 * fx) - abs(100 * fy));

    ptr[offset].x = red;
    ptr[offset].y = 0;
    ptr[offset].z = 0;
    ptr[offset].w = 255;
}

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
    return prev * (1 - alpha) + color * alpha;
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

__global__ void calculateNormals(unsigned char* rmnData, dim3 dataSize, float3* normals)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    float3 m, p, normal;
    dim3 s;

    s.x = dataSize.y * dataSize.z;
    s.y = dataSize.z;
    s.z = 1;

    if (i >= dataSize.x || j >= dataSize.y) return;

    for (size_t k = 0; k < dataSize.z; k++)
    {
        m.x = i == 0 ? 0 : rmnData[(i - 1) * s.x + (j - 0) * s.y + (k - 0) * s.z];
        p.x = i == dataSize.x - 1 ? 0 : rmnData[(i + 1) * s.x + (j + 0) * s.y + (k + 0) * s.z];
        normal.x = p.x - m.x;

        m.y = j == 0 ? 0 : rmnData[(i - 0) * s.x + (j - 1) * s.y + (k - 0) * s.z];
        p.y = j == dataSize.y - 1 ? 0 : rmnData[(i + 0) * s.x + (j + 1) * s.y + (k + 0) * s.z];
        normal.y = p.y - m.y;

        m.z = k == 0 ? 0 : rmnData[(i - 0) * s.x + (j - 0) * s.y + (k - 1) * s.z];
        p.z = k == dataSize.z - 1 ? 0 : rmnData[(i + 0) * s.x + (j + 0) * s.y + (k + 1) * s.z];
        normal.z = p.z - m.z;

        normals[i * s.x + j * s.y + k * s.z] = normalizeVector(normal);
    }
}

__device__ float meanPointValue(unsigned char* rmnData, dim3 dataSize, float3 point)
{
    int i = point.x;
    int j = point.y;
    int k = point.z;

    float xf = point.x - i;
    float yf = point.y - j;
    float zf = point.z - k;

    unsigned char c000, c001, c010, c011, c100, c101, c110, c111;

    if (i >= 0 && i + 1 < dataSize.x && j >= 0 && j + 1 < dataSize.y && k >= 0 && k + 1 < dataSize.z)
    {
        c000 = rmnData[(i + 0) * dataSize.y * dataSize.z + (j + 0) * dataSize.z + (k + 0)];
        c001 = rmnData[(i + 0) * dataSize.y * dataSize.z + (j + 0) * dataSize.z + (k + 1)];
        c010 = rmnData[(i + 0) * dataSize.y * dataSize.z + (j + 1) * dataSize.z + (k + 0)];
        c011 = rmnData[(i + 0) * dataSize.y * dataSize.z + (j + 1) * dataSize.z + (k + 1)];
        c100 = rmnData[(i + 1) * dataSize.y * dataSize.z + (j + 0) * dataSize.z + (k + 0)];
        c101 = rmnData[(i + 1) * dataSize.y * dataSize.z + (j + 0) * dataSize.z + (k + 1)];
        c110 = rmnData[(i + 1) * dataSize.y * dataSize.z + (j + 1) * dataSize.z + (k + 0)];
        c111 = rmnData[(i + 1) * dataSize.y * dataSize.z + (j + 1) * dataSize.z + (k + 1)];
    }

    float c00 = c000 * (1.0f - xf) + c100 * xf;
    float c10 = c010 * (1.0f - xf) + c110 * xf;
    float c01 = c001 * (1.0f - xf) + c101 * xf;
    float c11 = c011 * (1.0f - xf) + c111 * xf;

    float c0 = c00 * (1.0f - yf) + c10 * yf;
    float c1 = c01 * (1.0f - yf) + c11 * yf;

    float c = c0 * (1.0f - zf) + c1 * zf;

    return c;
}

__global__ void renderFrame(unsigned char* rmnData, dim3 dataSize, uint2 imageDim, unsigned int rotation, float3* normals, uchar4* ptr)
{
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t imageWidth = imageDim.x;
    size_t imageHeight = imageDim.y;
    size_t offset = (imageHeight - y - 1) * imageWidth + imageWidth - x - 1;

    if (x > imageWidth || y > imageHeight) return;

    float dsx = dataSize.x, dsy = dataSize.y, dsz = dataSize.z;

    float4 plane[6] = {
        { 1.0f, 0.0f, 0.0f,  0.0f },
        { 1.0f, 0.0f, 0.0f, -dsx  },
        { 0.0f, 1.0f, 0.0f,  0.0f },
        { 0.0f, 1.0f, 0.0f, -dsy  },
        { 0.0f, 0.0f, 1.0f,  0.0f },
        { 0.0f, 0.0f, 1.0f, -dsz  },
    };

    float3 viewPointPosition;
    float rmnDataDiagonalLength = length(dataSize.x, dataSize.y, dataSize.z);
    viewPointPosition.x = 0.5f * dataSize.x + 3 * rmnDataDiagonalLength * sin(rotation * pi / 180);
    viewPointPosition.y = -0.75f * dataSize.y;// +3 * rmnDataDiagonalLength * sin(90 * pi / 180);
    viewPointPosition.z = 0.5f * dataSize.z + 3 * rmnDataDiagonalLength * cos(rotation * pi / 180);// +3 * rmnDataDiagonalLength * cos(90 * pi / 180);

    float3 viewPointDirection;
    viewPointDirection.x = 0.5f * dataSize.x - viewPointPosition.x;
    viewPointDirection.y = 0.5f * dataSize.y - viewPointPosition.y;
    viewPointDirection.z = 0.5f * dataSize.z - viewPointPosition.z;
    viewPointDirection = normalizeVector(viewPointDirection);

    float3 jVersor = { 0.0f, 1.0f, 0.0f };
    float3 viewPointParalel;
    viewPointParalel.x = viewPointDirection.y * jVersor.z - viewPointDirection.z * jVersor.y;
    viewPointParalel.y = viewPointDirection.z * jVersor.x - viewPointDirection.x * jVersor.z;
    viewPointParalel.z = viewPointDirection.x * jVersor.y - viewPointDirection.y * jVersor.x;
    viewPointParalel = normalizeVector(viewPointParalel);

    float3 viewPointUp;
    viewPointUp.x = viewPointParalel.y * viewPointDirection.z - viewPointParalel.z * viewPointDirection.y;
    viewPointUp.y = viewPointParalel.z * viewPointDirection.x - viewPointParalel.x * viewPointDirection.z;
    viewPointUp.z = viewPointParalel.x * viewPointDirection.y - viewPointParalel.y * viewPointDirection.x;
    viewPointUp = normalizeVector(viewPointUp);

    /* Light position */
    float3 lightPosition;
    lightPosition.x = 0.5f * dataSize.x;
    lightPosition.y = -2.5f * dataSize.y;
    lightPosition.z = 0.5f * dataSize.z;
    lightPosition.x = -viewPointPosition.x;
    lightPosition.y = -viewPointPosition.y;
    lightPosition.z = -viewPointPosition.z;

    float3 viewPlane = { rmnDataDiagonalLength, rmnDataDiagonalLength, 0 };
    viewPlane.z = length(0.5f * dataSize.x - viewPointPosition.x, 0.5f * dataSize.y - viewPointPosition.y, 0.5f * dataSize.z - viewPointPosition.z);

    float3 point0 = viewPointPosition;
    float3 point1;

    float3 rayDirection;
    rayDirection.x = imageCoordonateToViewPlane(x, imageWidth, viewPlane.x);
    rayDirection.y = imageCoordonateToViewPlane(y, imageHeight, viewPlane.y);
    rayDirection.z = viewPlane.z;

    point1.x = viewPointDirection.x * rayDirection.z + viewPointUp.x * rayDirection.y + viewPointParalel.x * rayDirection.x;
    point1.y = viewPointDirection.y * rayDirection.z + viewPointUp.y * rayDirection.y + viewPointParalel.y * rayDirection.x;
    point1.z = viewPointDirection.z * rayDirection.z + viewPointUp.z * rayDirection.y + viewPointParalel.z * rayDirection.x;
    point1 = normalizeVector(point1);

    int st = 0;
    bool foundIntersection = false;
    float tmin = 0.0f, tmax = 0.0f;
    for (size_t k = 0; k < 6; k++)
    {
        /* Intersection between a line and a plane using parametric equation */
        /* Line d : x - x1 / a = y - y1 / b = z - z1 / c and the plane P : Ax + By + Cz + D = 0; */
        /* t = - Ax1 + By1 + Cz1 + D / Aa +Bb + Cc x|y|z = x1|y1|z1 - a|b|c * t */
        float numerator = plane[k].x * point0.x + plane[k].y * point0.y + plane[k].z * point0.z + plane[k].w;
        float denominator = plane[k].x * point1.x + plane[k].y * point1.y + plane[k].z * point1.z;

        /* Ray is parallel or coplanar with the plane, do nothing and move on */
        if (denominator == 0)
        {
            st = st < 1 ? 1 : st;
            continue;
        }

        float t = -numerator / denominator;

        //ts[6 * x + 6 * y * imageWidth + k] = t;

        /* The intersection is behind the camera */
        if (t < 0)
        {
            st = st < 2 ? 2 : st;
            continue;
        }

        float3 intersectionPoint;
        intersectionPoint.x = point0.x + point1.x * t;
        intersectionPoint.y = point0.y + point1.y * t;
        intersectionPoint.z = point0.z + point1.z * t;

        /* Intersection outside the dataset, do nothing and move on */
        if (intersectionPoint.x < -epsilon || intersectionPoint.x > dataSize.x + epsilon ||
            intersectionPoint.y < -epsilon || intersectionPoint.y > dataSize.y + epsilon ||
            intersectionPoint.z < -epsilon || intersectionPoint.z > dataSize.z + epsilon)
        {
            st = st < 3 ? 3 : st;
            continue;
        }

        if (foundIntersection)
        {
            tmin = tmin > t ? t : tmin;
            tmax = tmax < t ? t : tmax;
        }
        else
        {
            foundIntersection = true;
            tmin = t;
            tmax = t;
        }
    }

    /* No intersection found, set the color to black */
    if (!foundIntersection)
    {
        ptr[offset].x = 0;
        ptr[offset].y = 0;
        ptr[offset].z = 0;
        ptr[offset].w = 255;

        return;
    }

    float3 prevNormal = { 0, 0, 0 }, point, normal, light;
    float nlcos, color[3] = { 0, 0, 0 }, c[4], value;
    int position;

    for (float t = tmax; t >= tmin; t -= step)
    {
        point.x = point0.x + point1.x * t;
        point.y = point0.y + point1.y * t;
        point.z = point0.z + point1.z * t;

        value = meanPointValue(rmnData, dataSize, point);
        position = getPositionForColormapEntryValue(value);
        if (position < 0) continue;

        c[R] = getColorValue(position, R);
        c[G] = getColorValue(position, G);
        c[B] = getColorValue(position, B);
        c[A] = getColorValue(position, A);

        position = pointNormal(dataSize, point);

        normal.x = normals[position].x;
        normal.y = normals[position].y;
        normal.z = normals[position].z;

        if (fabsf(normal.x) <= epsilon && fabsf(normal.y) <= epsilon && fabsf(normal.z) <= epsilon)
        {
            normal.x = prevNormal.x;
            normal.y = prevNormal.y;
            normal.z = prevNormal.z;
        }
        else
        {
            prevNormal.x = normal.x;
            prevNormal.y = normal.y;
            prevNormal.z = normal.z;
        }

        light.x = lightPosition.x - point.x;
        light.y = lightPosition.y - point.y;
        light.z = lightPosition.z - point.z;
        light = normalizeVector(light);

        nlcos = normal.x * light.x + normal.y * light.y + normal.z * light.z;

        c[R] *= ambient;
        c[G] *= ambient;
        c[B] *= ambient;

        if (nlcos > 0)
        {
            c[R] += c[R] * diffuse * nlcos;
            c[G] += c[G] * diffuse * nlcos;
            c[B] += c[B] * diffuse * nlcos;
        }

        color[R] = composeRGBA(color[R], c[R], c[A]);
        color[G] = composeRGBA(color[G], c[G], c[A]);
        color[B] = composeRGBA(color[B], c[B], c[A]);
    }

    ptr[offset].x = colorFloatToByte(color[R]);
    ptr[offset].y = colorFloatToByte(color[G]);
    ptr[offset].z = colorFloatToByte(color[B]);
    ptr[offset].w = 255;
}

typedef struct _KernelLaunchParams
{
    unsigned char* dev_rmnData;
    float3* dev_normals;
    uint2 imageDim;
    dim3 rmnDim;
    unsigned int rotation;
}KernelLaunchParams;

cudaEvent_t start, stop;

__host__ void renderFrame(uchar4* pixels, void* parameters, size_t ticks)
{
    KernelLaunchParams* kernelParams = static_cast<KernelLaunchParams*>(parameters);

    dim3 threads(8, 16);
    dim3 grids(imageHeigth / threads.x + 1, imageWidth / threads.y + 1);

    cudaEventRecord(start);

    renderFrame << <grids, threads >> > (kernelParams->dev_rmnData, kernelParams->rmnDim, kernelParams->imageDim, kernelParams->rotation, kernelParams->dev_normals, pixels);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    float fps = 1000 / milliseconds;
    fpsDisplay.displayFps(pixels, fps);
    
    kernelParams->rotation += 1;
    kernelParams->rotation %= 360;
}

int main(int argc, char** argv)
{
    RmnDatasetFileLoader rmnDatasetFileLoader("../Data", "vertebra");

    unsigned char* dev_rmnData;
    float3* dev_normals;
    cudaError cudaError;

    uint2 imageDim;
    imageDim.x = imageHeigth;
    imageDim.y = imageWidth;

    try
    {
        cudaError = cudaEventCreate(&start);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaEventCreate", cudaError, __FILE__, __LINE__));

        cudaError = cudaEventCreate(&stop);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaEventCreate", cudaError, __FILE__, __LINE__));

        rmnDatasetFileLoader.loadDataset();

        for (size_t c = 0; c < rmnDatasetFileLoader.getColor().size(); c++)
        {
            setColorValue << <1, 1 >> > (c, R, rmnDatasetFileLoader.getColor()[c].r);
            setColorValue << <1, 1 >> > (c, G, rmnDatasetFileLoader.getColor()[c].g);
            setColorValue << <1, 1 >> > (c, B, rmnDatasetFileLoader.getColor()[c].b);
            setColorValue << <1, 1 >> > (c, A, rmnDatasetFileLoader.getColor()[c].a);
        }

        for (size_t c = 0; c < rmnDatasetFileLoader.getColormap().size(); c++)
        {
            if (c % 2 == 0)
                setColormapValue << <1, 1 >> > (c / 2, rmnDatasetFileLoader.getColormap()[c]);
            else
                setColormapColor << <1, 1 >> > (c / 2, rmnDatasetFileLoader.getColormap()[c]);
        }

        rmnDim = rmnDatasetFileLoader.getRmnDatasetDimensions();

        cudaError = cudaMalloc((void**)&dev_rmnData, rmnDim.x * rmnDim.y * rmnDim.z * sizeof(char));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));

        cudaError = cudaMemcpy(dev_rmnData, rmnDatasetFileLoader.getRmnDataset(), rmnDim.x * rmnDim.y * rmnDim.z * sizeof(char), cudaMemcpyHostToDevice);
        if (cudaError != cudaError::cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpy", cudaError, __FILE__, __LINE__));

        cudaError = cudaMalloc((void**)&dev_normals, rmnDim.x * rmnDim.y * rmnDim.z * sizeof(float3));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));

        dim3 grids(rmnDim.x / 16 + 1, rmnDim.y / 16 + 1);
        dim3 threads(16, 16);

        calculateNormals << <grids, threads >> > (dev_rmnData, rmnDim, dev_normals);
        cudaDeviceSynchronize();

        KernelLaunchParams params;
        params.dev_normals = dev_normals;
        params.dev_rmnData = dev_rmnData;
        params.imageDim = imageDim;
        params.rmnDim = rmnDim;
        params.rotation = 0;

        GpuGLAnim::animAdExit(renderFrame, nullptr, imageHeigth, imageWidth, static_cast<void*>(&params));
    }
    catch (exception& ex)
    {
        cerr << ex.what() << endl;
    }
    catch (...)
    {
        cerr << "Fatal error!" << endl;
    }
    
    return 0;
}