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
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#define R 0
#define G 1
#define B 2
#define A 3

#define pi 3.141592653f
#define epsilon 0.001f
#define step 0.5f
#define ambient 0.3f
#define diffuse 0.7f

using std::cout;
using std::cerr;
using std::endl;
using std::getline;
using std::exception;
using std::function;
using std::string;
using std::runtime_error;
using std::unique_ptr;
using std::vector;

dim3 rmnDim;

unsigned int imageHeigth = 512;
unsigned int imageWidth = 512;

const int dim = 512;

FpsDisplay fpsDisplay({imageHeigth, imageWidth});


float* dev_r;
float* dev_g;
float* dev_b;
float* dev_a;

texture<float> tex_r;
texture<float> tex_g;
texture<float> tex_b;
texture<float> tex_a;


__constant__ float4 plane[6];


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

__global__ void renderFrame(unsigned char* rmnData, dim3 dataSize, uint2 imageDim, unsigned int rotation, float3* normals, uchar4* ptr, float* r, float* g, float* b, float* a)
{
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t imageWidth = imageDim.x;
    size_t imageHeight = imageDim.y;
    size_t offset = (imageHeight - y - 1) * imageWidth + imageWidth - x - 1;

    if (x > imageWidth || y > imageHeight) return;

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

        /* Ray is parallel or coplanar with the plane, do nothing and move on */
        if (plane[k].x * point1.x + plane[k].y * point1.y + plane[k].z * point1.z == 0)
        {
            st = st < 1 ? 1 : st;
            continue;
        }

        float t = -(plane[k].x * point0.x + plane[k].y * point0.y + plane[k].z * point0.z + plane[k].w) / (plane[k].x * point1.x + plane[k].y * point1.y + plane[k].z * point1.z);

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

    float3 prevNormal = { 0, 0, 0 }, point, normal, light, lastNormal;
    float color[3] = { 0, 0, 0 }, c[4];
    int position, lastPosition = 0;

    for (float t = tmax; t >= tmin; t -= step)
    {
        point.x = point0.x + point1.x * t;
        point.y = point0.y + point1.y * t;
        point.z = point0.z + point1.z * t;

        //position = getPositionForColormapEntryValue(meanPointValue(rmnData, dataSize, point));
        //if (position < 0) continue;

        position = meanPointValue(rmnData, dataSize, point);

        c[R] = tex1Dfetch(tex_r, position);
        c[G] = tex1Dfetch(tex_g, position);
        c[B] = tex1Dfetch(tex_b, position);
        c[A] = tex1Dfetch(tex_a, position);

        position = pointNormal(dataSize, point);

        /* COMMENT HERE TO REMOVE REGISTERY CACHING*/
        if (position != lastPosition)
        {
          normal.x = normals[position].x;
          normal.y = normals[position].y;
          normal.z = normals[position].z;
        
            lastPosition = position;
        }
        else
        {
            normal = lastNormal;
        }

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

        //nlcos = normal.x * light.x + normal.y * light.y + normal.z * light.z;

        c[R] *= ambient;
        c[G] *= ambient;
        c[B] *= ambient;

        if (normal.x * light.x + normal.y * light.y + normal.z * light.z > 0)
        {
            c[R] += c[R] * diffuse * (normal.x * light.x + normal.y * light.y + normal.z * light.z);
            c[G] += c[G] * diffuse * (normal.x * light.x + normal.y * light.y + normal.z * light.z);
            c[B] += c[B] * diffuse * (normal.x * light.x + normal.y * light.y + normal.z * light.z);
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
float minFps = 1000, meanFps = 0;

__host__ void renderFrame(uchar4* pixels, void* parameters, size_t ticks)
{
    KernelLaunchParams* kernelParams = static_cast<KernelLaunchParams*>(parameters);

    dim3 threads(8, 16);
    dim3 grids(imageHeigth / threads.x + 1, imageWidth / threads.y + 1);

    cudaEventRecord(start);

    renderFrame << <grids, threads >> > (kernelParams->dev_rmnData, kernelParams->rmnDim, kernelParams->imageDim, kernelParams->rotation, kernelParams->dev_normals, pixels, dev_r, dev_g, dev_b, dev_a);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float fps = 1000 / milliseconds;
    fpsDisplay.displayFps(pixels, fps);

    static float f[360];
    static int i = 0;

    meanFps += fps;
    if (fps < minFps)
        minFps = fps;

    f[i] = fps; i++;

    kernelParams->rotation += 1;
    if (kernelParams->rotation == 359)
    {
        float stdDev = 0;

        meanFps = meanFps / 360;

        for (int k = 0; k < 360; k++)
            stdDev += abs(f[k] - meanFps);

        cout << "Minimum fps " << minFps << endl;
        cout << "Mean fps " << meanFps << endl;
        cout << "Standard dev fps " << stdDev / 360 << endl;
        exit(0);
    }
}

unique_ptr<unsigned char, function<void(unsigned char*)>> dev_normals_uptr;
unique_ptr<unsigned char, function<void(unsigned char*)>> dev_rmnData_uptr;

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

        //cudaError = cudaMemcpyToSymbol(colors, rmnDatasetFileLoader.getColor().data(), rmnDatasetFileLoader.getColor().size() * sizeof(Color));
        //if (cudaError != cudaSuccess)
        //    throw runtime_error(makeCudaErrorMessage("cudaMemcpyToSymbol", cudaError, __FILE__, __LINE__));

        //cudaError = cudaMemcpyToSymbol(colormap, host_colormap, 256 * sizeof(unsigned int));
        //if (cudaError != cudaSuccess)
        //    throw runtime_error(makeCudaErrorMessage("cudaMemcpyToSymbol", cudaError, __FILE__, __LINE__));
        //
        //cudaError = cudaMemcpyToSymbol(colormapLength, &colormapSize, sizeof(size_t));
        //if (cudaError != cudaSuccess)
        //    throw runtime_error(makeCudaErrorMessage("cudaMemcpyToSymbol", cudaError, __FILE__, __LINE__));

        size_t colormapSize = rmnDatasetFileLoader.getColormap().size();
        unsigned int host_colormap[256];

        for (size_t i = 0; i < 256; i++)
        {
            for (size_t j = 0; j < colormapSize; j += 2)
            {
                if (i < rmnDatasetFileLoader.getColormap()[j+2])
                {
                    host_colormap[i] = j / 2;
                    break;
                }
            }
        }

        float host_r[256], host_g[256], host_b[256], host_a[256];

        for (size_t i = 0; i < 256; i++)
        {
            host_r[i] = rmnDatasetFileLoader.getColor()[host_colormap[i]].r;
            host_g[i] = rmnDatasetFileLoader.getColor()[host_colormap[i]].g;
            host_b[i] = rmnDatasetFileLoader.getColor()[host_colormap[i]].b;
            host_a[i] = rmnDatasetFileLoader.getColor()[host_colormap[i]].a;
        }

        cudaError = cudaMalloc((void**)&dev_r, 256 * sizeof(float));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));

        cudaError = cudaMalloc((void**)&dev_g, 256 * sizeof(float));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));

        cudaError = cudaMalloc((void**)&dev_b, 256 * sizeof(float));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));

        cudaError = cudaMalloc((void**)&dev_a, 256 * sizeof(float));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));


        cudaError = cudaMemcpy(dev_r, host_r, 256 * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpy", cudaError, __FILE__, __LINE__));
        
        cudaError = cudaMemcpy(dev_g, host_g, 256 * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpy", cudaError, __FILE__, __LINE__));

        cudaError = cudaMemcpy(dev_b, host_b, 256 * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpy", cudaError, __FILE__, __LINE__));

        cudaError = cudaMemcpy(dev_a, host_a, 256 * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpy", cudaError, __FILE__, __LINE__));

        cudaChannelFormatDesc descrFloat = cudaCreateChannelDesc<float>();

        cudaError = cudaBindTexture(0, tex_r, dev_r, descrFloat, 256);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaBindTexture", cudaError, __FILE__, __LINE__));

        cudaError = cudaBindTexture(0, tex_g, dev_g, descrFloat, 256);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaBindTexture", cudaError, __FILE__, __LINE__));

        cudaError = cudaBindTexture(0, tex_b, dev_b, descrFloat, 256);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaBindTexture", cudaError, __FILE__, __LINE__));

        cudaError = cudaBindTexture(0, tex_a, dev_a, descrFloat, 256);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaBindTexture", cudaError, __FILE__, __LINE__));

        rmnDim = rmnDatasetFileLoader.getRmnDatasetDimensions();

        float dsx = rmnDim.x, dsy = rmnDim.y, dsz = rmnDim.z;

        float4 planeHost[6] = {
            { 1.0f, 0.0f, 0.0f,  0.0f },
            { 1.0f, 0.0f, 0.0f, -dsx },
            { 0.0f, 1.0f, 0.0f,  0.0f },
            { 0.0f, 1.0f, 0.0f, -dsy },
            { 0.0f, 0.0f, 1.0f,  0.0f },
            { 0.0f, 0.0f, 1.0f, -dsz },
        };

        cudaError = cudaMemcpyToSymbol(plane, planeHost, 6 * sizeof(float4));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpyToSymbol", cudaError, __FILE__, __LINE__));


        cudaError = cudaMalloc((void**)&dev_rmnData, rmnDim.x * rmnDim.y * rmnDim.z * sizeof(char));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));

        dev_rmnData_uptr = unique_ptr<unsigned char, function<void(unsigned char*)>>(
            dev_rmnData, 
            [](unsigned char* dev_ptr) { cout << "CudaFree" << endl; cudaFree(dev_ptr); }
        );

        cudaError = cudaMemcpy(dev_rmnData, rmnDatasetFileLoader.getRmnDataset(), rmnDim.x * rmnDim.y * rmnDim.z * sizeof(char), cudaMemcpyHostToDevice);
        if (cudaError != cudaError::cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpy", cudaError, __FILE__, __LINE__));

        cudaError = cudaMalloc((void**)&dev_normals, rmnDim.x * rmnDim.y * rmnDim.z * sizeof(float3));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));

        dev_normals_uptr = unique_ptr<unsigned char, function<void(unsigned char*)>> (
            dev_rmnData,
            [](unsigned char* dev_ptr) { cout << "CudaFree" << endl; cudaFree(dev_ptr); }
        );

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