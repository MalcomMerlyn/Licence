#include <GL/glut.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"

#include "GpuGlInteropAnim.h"
#include "RmnDatasetFileLoader.h"
#include "FpsDisplay.h"
#include "KernelLaunchParams.h"
#include "RayCastUtils.h"

#include <stdio.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

using std::cout;
using std::cerr;
using std::endl;
using std::getline;
using std::exception;
using std::function;
using std::max;
using std::string;
using std::runtime_error;
using std::unique_ptr;
using std::vector;

__device__ const unsigned int R = 0;
__device__ const unsigned int G = 1;
__device__ const unsigned int B = 2;
__device__ const unsigned int A = 3;

__device__ const float pi = 3.141592653f;
__device__ const float epsilon = 0.001f;
__device__ const float step = 0.5f;
__device__ const float ambient = 0.3f;
__device__ const float diffuse = 0.7f;

unsigned int imageHeigth = 512;
unsigned int imageWidth = 512;

texture<char, 3, cudaReadModeElementType> textureRmnData;
cudaArray* dev_rmnDataArray = 0;

FpsDisplay fpsDisplay({ imageHeigth, imageWidth });

float* dev_r;
float* dev_g;
float* dev_b;
float* dev_a;

__constant__ float4 plane[6];

__global__ void calculateNormals(unsigned char* rmnData, dim3 dataSize, float3* normals)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    float3 m, p, normal;
    dim3 s;

    int off[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    s.x = dataSize.y * dataSize.z;
    s.y = dataSize.z;
    s.z = 1;

    const float mul = 0.4f;

    if (i == 0 || j == 0 || i >= dataSize.x - 1 || j >= dataSize.y - 1) return;

    for (size_t k = 1; k < dataSize.z - 1; k++)
    {
        // Add the cell above and below
        m.x = i == 0 ? 0 : rmnData[(i - 1) * s.x + (j - 0) * s.y + (k - 0) * s.z];
        p.x = i == dataSize.x - 1 ? 0 : rmnData[(i + 1) * s.x + (j + 0) * s.y + (k + 0) * s.z];

        m.y = j == 0 ? 0 : rmnData[(i - 0) * s.x + (j - 1) * s.y + (k - 0) * s.z];
        p.y = j == dataSize.y - 1 ? 0 : rmnData[(i + 0) * s.x + (j + 1) * s.y + (k + 0) * s.z];

        m.z = k == 0 ? 0 : rmnData[(i - 0) * s.x + (j - 0) * s.y + (k - 1) * s.z];
        p.z = k == dataSize.z - 1 ? 0 : rmnData[(i + 0) * s.x + (j +0) * s.y + (k + 1) * s.z];

        // Add the second cell above and below
        m.x += i - 2 < 0 ? 0 : rmnData[(i - 2) * s.x + (j - 0) * s.y + (k - 0) * s.z];
        p.x += i + 2 == dataSize.x ? 0 : rmnData[(i + 2) * s.x + (j + 0) * s.y + (k + 0) * s.z];
            
        m.y += j - 2 < 0 ? 0 : rmnData[(i - 0) * s.x + (j - 2) * s.y + (k - 0) * s.z];
        p.y += j + 2 == dataSize.y ? 0 : rmnData[(i + 0) * s.x + (j + 2) * s.y + (k + 0) * s.z];
            
        m.z += k - 2 < 0 ? 0 : rmnData[(i - 0) * s.x + (j - 0) * s.y + (k - 2) * s.z];
        p.z += k + 2 > dataSize.z ? 0 : rmnData[(i + 0) * s.x + (j + 0) * s.y + (k + 2) * s.z];

        // Add the cells sorounding the cell above and bellow with factor mul
        for (int o = 0; o < 9; o++)
        {
            m.x += mul * (i == 0 ? 0 : rmnData[(i - 1) * s.x + (j - off[o][0]) * s.y + (k - off[o][1]) * s.z]);
            p.x += mul * (i == dataSize.x - 1 ? 0 : rmnData[(i + 1) * s.x + (j + off[o][0]) * s.y + (k + off[o][1]) * s.z]);
                         
            m.y += mul * (j == 0 ? 0 : rmnData[(i - off[o][0]) * s.x + (j - 1) * s.y + (k - off[o][1]) * s.z]);
            p.y += mul * (j == dataSize.y - 1 ? 0 : rmnData[(i + off[o][0]) * s.x + (j + 1) * s.y + (k + off[o][1]) * s.z]);
                         
            m.z += mul * (k == 0 ? 0 : rmnData[(i - off[o][0]) * s.x + (j - off[o][1]) * s.y + (k - 1) * s.z]);
            p.z += mul * (k == dataSize.z - 1 ? 0 : rmnData[(i + off[o][0]) * s.x + (j + off[o][1]) * s.y + (k + 1) * s.z]);
        }

        normal.x = p.x - m.x;
        normal.y = p.y - m.y;
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

    c000 = tex3D(textureRmnData, i + 0, j + 0, k + 0);
    c001 = tex3D(textureRmnData, i + 0, j + 0, k + 1);
    c010 = tex3D(textureRmnData, i + 0, j + 1, k + 0);
    c011 = tex3D(textureRmnData, i + 0, j + 1, k + 1);
    c100 = tex3D(textureRmnData, i + 1, j + 0, k + 0);
    c101 = tex3D(textureRmnData, i + 1, j + 0, k + 1);
    c110 = tex3D(textureRmnData, i + 1, j + 1, k + 0);
    c111 = tex3D(textureRmnData, i + 1, j + 1, k + 1);
    
    float c00 = c000 * (1.0f - xf) + c100 * xf;
    float c10 = c010 * (1.0f - xf) + c110 * xf;
    float c01 = c001 * (1.0f - xf) + c101 * xf;
    float c11 = c011 * (1.0f - xf) + c111 * xf;

    float c0 = c00 * (1.0f - yf) + c10 * yf;
    float c1 = c01 * (1.0f - yf) + c11 * yf;

    float c = c0 * (1.0f - zf) + c1 * zf;

    return c;
}

__global__ void renderFrame(unsigned char* rmnData, dim3 dataSize, uint2 imageDim, int2 rotation, float3* normals, uchar4* ptr, float* r, float* g, float* b, float* a, float ratio)
{
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t imageWidth = imageDim.x;
    size_t imageHeight = imageDim.y;
    size_t offset = (imageHeight - y - 1) * imageWidth + imageWidth - x - 1;

    if (x > imageWidth || y > imageHeight) return;

    float3 viewPointPosition;
    float rmnDataDiagonalLength = length(dataSize.x, dataSize.y, dataSize.z);
    viewPointPosition.x = ratio * rmnDataDiagonalLength * sin(rotation.x * pi / 180) * cos(rotation.y * pi / 180);
    viewPointPosition.y = ratio * rmnDataDiagonalLength * cos(rotation.x * pi / 180);
    viewPointPosition.z = ratio * rmnDataDiagonalLength * sin(rotation.x * pi / 180) * sin(rotation.y * pi / 180);

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

        c[R] = r[position];
        c[G] = g[position];
        c[B] = b[position];
        c[A] = a[position];

        if (c[A] < 0.1) continue;

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

cudaEvent_t start, stop;
float minFps = 1000, meanFps = 0;

__host__ void renderFrame(uchar4* pixels, void* parameters, size_t ticks)
{
    KernelLaunchParams* kernelParams = static_cast<KernelLaunchParams*>(parameters);

    dim3 threads(8, 16);
    dim3 grids(imageHeigth / threads.x + 1, imageWidth / threads.y + 1);

    cudaEventRecord(start);

    renderFrame << <grids, threads >> > (
        kernelParams->dev_rmnData
        , kernelParams->rmnDim
        , kernelParams->imageDim
        , kernelParams->rotation
        , kernelParams->dev_normals
        , pixels
        , dev_r
        , dev_g
        , dev_b
        , dev_a
        , kernelParams->ratio
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float fps = 1000 / milliseconds;
    fpsDisplay.displayFps(pixels, fps);

    //static float f[360];
    //static int i = 0;

    //meanFps += fps;
    //if (fps < minFps && kernelParams->rotation > 5) 
    //    minFps = fps;

    //f[i] = fps; i++;

    //kernelParams->rotation += 1;
    //if (kernelParams->rotation == 359)
    //{
    //    float stdDev = 0;
    //
    //    meanFps = meanFps / 360;
    //
    //    for (int k = 0; k < 360; k++)
    //        stdDev += abs(f[k] - meanFps);
    //
    //    cout << "Minimum fps " << minFps << endl;
    //    cout << "Mean fps " << meanFps << endl;
    //    cout << "Standard dev fps " << stdDev / 360 << endl;
    //    cudaDeviceReset();
    //    exit(0);
    //}
}

unique_ptr<unsigned char, function<void(unsigned char*)>> dev_normals_uptr;
unique_ptr<unsigned char, function<void(unsigned char*)>> dev_rmnData_uptr;
unique_ptr<unsigned char, function<void(unsigned char*)>> dev_rmnDataUnaligned_uptr;


int main(int argc, char** argv)
{
    RmnDatasetFileLoader rmnDatasetFileLoader(argv[1], argv[2]);

    imageWidth = atoi(argv[3]);
    imageHeigth = atoi(argv[4]);

    printf("%d %d\n", imageWidth, imageHeigth);

    dim3 rmnDim;

    unsigned char* dev_rmnData;
    unsigned char* dev_rmnDataUnaligned;
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
            throw runtime_error(makeCudaErrorMessage("cudaMemcpyToSymbol", cudaError, __FILE__, __LINE__));
        
        cudaError = cudaMemcpy(dev_g, host_g, 256 * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpyToSymbol", cudaError, __FILE__, __LINE__));

        cudaError = cudaMemcpy(dev_b, host_b, 256 * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpyToSymbol", cudaError, __FILE__, __LINE__));

        cudaError = cudaMemcpy(dev_a, host_a, 256 * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpyToSymbol", cudaError, __FILE__, __LINE__));


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


        cudaError = cudaMalloc((void**)&dev_rmnDataUnaligned, rmnDim.x * rmnDim.y * rmnDim.z * sizeof(char));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));

        dev_rmnDataUnaligned_uptr = unique_ptr<unsigned char, function<void(unsigned char*)>>(
            dev_rmnDataUnaligned, 
            [](unsigned char* dev_ptr) { cout << "CudaFree" << endl; cudaFree(dev_ptr); }
        );

        cudaError = cudaMemcpy(dev_rmnDataUnaligned, rmnDatasetFileLoader.getRmnDataset(), rmnDim.x * rmnDim.y * rmnDim.z * sizeof(char), cudaMemcpyHostToDevice);
        if (cudaError != cudaError::cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpy", cudaError, __FILE__, __LINE__));

        cudaError = cudaMalloc((void**)&dev_normals, rmnDim.x * rmnDim.y * rmnDim.z * sizeof(float3));
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc", cudaError, __FILE__, __LINE__));

        dev_normals_uptr = unique_ptr<unsigned char, function<void(unsigned char*)>> (
            (unsigned char*)dev_normals,
            [](unsigned char* dev_ptr) { cout << "CudaFree" << endl; cudaFree(dev_ptr); }
        );

        unsigned char* reorderedRmnData = reinterpret_cast<unsigned char*>(malloc(rmnDim.x * rmnDim.y * rmnDim.z * sizeof(unsigned char)));
        if (reorderedRmnData == nullptr)
            throw runtime_error(makeErrnoErrorMessage("malloc", __FILE__, __LINE__));

        for (size_t i = 0; i < rmnDim.x; i++)
            for (size_t j = 0; j < rmnDim.y; j++)
                for (size_t k = 0; k < rmnDim.z; k++)
                {
                    reorderedRmnData[k * rmnDim.x * rmnDim.y + j * rmnDim.x + i] =
                        rmnDatasetFileLoader.getRmnDataset()[i * rmnDim.y * rmnDim.z + j * rmnDim.z + k];
                }

        cudaChannelFormatDesc descriptor = cudaCreateChannelDesc<char>();
        cudaExtent rmnDataVolumeSize = { rmnDim.x, rmnDim.y, rmnDim.z };
        cudaError = cudaMalloc3DArray(&dev_rmnDataArray, &descriptor, rmnDataVolumeSize);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMalloc3DArray", cudaError, __FILE__, __LINE__));

        cudaMemcpy3DParms memcpyParams;
        memset(&memcpyParams, 0, sizeof(memcpyParams));
        memcpyParams.srcPtr = make_cudaPitchedPtr((void*)reorderedRmnData, rmnDataVolumeSize.width * sizeof(char), rmnDataVolumeSize.width, rmnDataVolumeSize.height);
        memcpyParams.dstArray = dev_rmnDataArray;
        memcpyParams.extent = rmnDataVolumeSize;
        memcpyParams.kind = cudaMemcpyHostToDevice;

        cudaError = cudaMemcpy3D(&memcpyParams);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpy3D", cudaError, __FILE__, __LINE__));

        textureRmnData.normalized = false;                      // access with normalized texture coordinates
        textureRmnData.filterMode = cudaFilterModePoint;      // linear interpolation 
        textureRmnData.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
        textureRmnData.addressMode[1] = cudaAddressModeWrap;
        textureRmnData.addressMode[2] = cudaAddressModeWrap;

        // bind array to 3D texture
        cudaError = cudaBindTextureToArray(textureRmnData, dev_rmnDataArray, descriptor);
        if (cudaError != cudaSuccess)
            throw runtime_error(makeCudaErrorMessage("cudaMemcpy3D", cudaError, __FILE__, __LINE__));

        dim3 grids(rmnDim.x / 16 + 1, rmnDim.y / 16 + 1);
        dim3 threads(16, 16);

        calculateNormals << <grids, threads >> > (dev_rmnDataUnaligned, rmnDim, dev_normals);
        cudaDeviceSynchronize();

        KernelLaunchParams params;
        params.dev_normals = dev_normals;
        params.dev_rmnData = dev_rmnData;
        params.imageDim = imageDim;
        params.rmnDim = rmnDim;
        //params.rotation = 0;

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