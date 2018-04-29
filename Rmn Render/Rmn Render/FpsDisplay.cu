#include "FpsDisplay.h"

#include <stdexcept>

using std::runtime_error;

__global__ static void drawDigitPixel(char* devDigits, uint2 digitDim, size_t digit, uint2 imageDim, uchar4* pixels)
{
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t imageWidth = imageDim.x;
    size_t imageHeight = imageDim.y;
    size_t offset = (imageHeight - y - 1) * imageWidth + imageWidth - x - 1;

    if (devDigits[digitDim.x - x - 1 + y * digitDim.x + digit * digitDim.x * digitDim.y] != 0)
    {
        pixels[offset].x = 255;
        pixels[offset].y = 0;
        pixels[offset].z = 0;
    }
}

__global__ static void drawPointPixel(char* devPoint, uint2 pointDim, uint2 imageDim, uchar4* pixels)
{
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t imageWidth = imageDim.x;
    size_t imageHeight = imageDim.y;
    size_t offset = (imageHeight - y - 1) * imageWidth + imageWidth - x - 1;

    if (devPoint[x + y * pointDim.x] != 0)
    {
        pixels[offset].x = 255;
        pixels[offset].y = 0;
        pixels[offset].z = 0;
    }
}

FpsDisplay::FpsDisplay(uint2 imageDim) : 
    m_imageDim{ imageDim.x, imageDim.y }
{
    cudaError error;

    error = cudaMalloc((void**)&m_devDigits, NumberOfDigits * DigitHeight * DigitWidth * sizeof(char));
    if (error != cudaSuccess)
        throw runtime_error(makeErrorMessage("cudaMalloc", error, __FILE__, __LINE__));

    error = cudaMemcpy(m_devDigits, m_digits, NumberOfDigits * DigitHeight * DigitWidth * sizeof(char), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        throw runtime_error(makeErrorMessage("cudaMemcpy", error, __FILE__, __LINE__));

    error = cudaMalloc((void**)&m_devPoint, DigitHeight * PointWidth * sizeof(char));
    if (error != cudaSuccess)
        throw runtime_error(makeErrorMessage("cudaMalloc", error, __FILE__, __LINE__));

    error = cudaMemcpy(m_devPoint, m_point, DigitHeight * PointWidth * sizeof(char), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        throw runtime_error(makeErrorMessage("cudaMemcpy", error, __FILE__, __LINE__));
}

void FpsDisplay::displayFps(uchar4* pixels, float fps)
{
    unsigned int fps100 = fps * 100;
    size_t digit, position = 2;
    
    digit = fps100 % 10;
    drawDigit(digit, m_imageDim, pixels);
    fps100 /= 10;

    digit = fps100 % 10;
    drawDigit(digit, m_imageDim, pixels - DigitWidth);
    fps100 /= 10;
    
    drawPoint(m_imageDim, pixels - 2 * DigitWidth);
    
    while (fps100 != 0)
    {
        digit = fps100 % 100;
        drawDigit(digit, m_imageDim, pixels - position * DigitWidth - PointWidth);
        fps100 /= 10;
        position++;
    }

    cudaDeviceSynchronize();
}

void FpsDisplay::drawDigit(size_t digit, uint2 imageDim, uchar4* pixels)
{
    dim3 threads(DigitWidth, DigitHeight);

    uint2 digitDim;
    digitDim.x = DigitWidth;
    digitDim.y = DigitHeight;

    drawDigitPixel << <1, threads >> > (m_devDigits, digitDim, digit, imageDim, pixels);
}

void FpsDisplay::drawPoint(uint2 imageDim, uchar4* pixels)
{
    dim3 threads(PointWidth, DigitHeight);

    uint2 pointDim;
    pointDim.x = PointWidth;
    pointDim.y = DigitHeight;

    drawPointPixel << <1, threads >> > (m_devPoint, pointDim, imageDim, pixels);
}
