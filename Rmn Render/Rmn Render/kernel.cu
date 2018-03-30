#include <GL/glut.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"

#include <iostream>

#ifndef GL_ARB_pixel_buffer_object
#define GL_PIXEL_PACK_BUFFER_ARB            0x88EB
#define GL_PIXEL_UNPACK_BUFFER_ARB          0x88EC
#define GL_PIXEL_PACK_BUFFER_BINDING_ARB    0x88ED
#define GL_PIXEL_UNPACK_BUFFER_BINDING_ARB  0x88EF
#endif

#ifndef GL_ARB_vertex_buffer_object
#define GL_BUFFER_SIZE_ARB                0x8764
#define GL_BUFFER_USAGE_ARB               0x8765
#define GL_ARRAY_BUFFER_ARB               0x8892
#define GL_ELEMENT_ARRAY_BUFFER_ARB       0x8893
#define GL_ARRAY_BUFFER_BINDING_ARB       0x8894
#define GL_ELEMENT_ARRAY_BUFFER_BINDING_ARB 0x8895
#define GL_VERTEX_ARRAY_BUFFER_BINDING_ARB 0x8896
#define GL_NORMAL_ARRAY_BUFFER_BINDING_ARB 0x8897
#define GL_COLOR_ARRAY_BUFFER_BINDING_ARB 0x8898
#define GL_INDEX_ARRAY_BUFFER_BINDING_ARB 0x8899
#define GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING_ARB 0x889A
#define GL_EDGE_FLAG_ARRAY_BUFFER_BINDING_ARB 0x889B
#define GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING_ARB 0x889C
#define GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING_ARB 0x889D
#define GL_WEIGHT_ARRAY_BUFFER_BINDING_ARB 0x889E
#define GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING_ARB 0x889F
#define GL_READ_ONLY_ARB                  0x88B8
#define GL_WRITE_ONLY_ARB                 0x88B9
#define GL_READ_WRITE_ARB                 0x88BA
#define GL_BUFFER_ACCESS_ARB              0x88BB
#define GL_BUFFER_MAPPED_ARB              0x88BC
#define GL_BUFFER_MAP_POINTER_ARB         0x88BD
#define GL_STREAM_DRAW_ARB                0x88E0
#define GL_STREAM_READ_ARB                0x88E1
#define GL_STREAM_COPY_ARB                0x88E2
#define GL_STATIC_DRAW_ARB                0x88E4
#define GL_STATIC_READ_ARB                0x88E5
#define GL_STATIC_COPY_ARB                0x88E6
#define GL_DYNAMIC_DRAW_ARB               0x88E8
#define GL_DYNAMIC_READ_ARB               0x88E9
#define GL_DYNAMIC_COPY_ARB               0x88EA
#endif

typedef void(*ProcGlGenBuffers)(GLsizei n, GLuint* buffers);
typedef void(*ProcGlBindBuffer)(GLenum target, GLuint buffer);
typedef void(*ProcGlBufferData)(GLenum target, GLsizei size, const GLvoid* data, GLenum usage);
typedef void(*ProcGlDeleteBuffers)(GLsizei n, const GLuint * buffers);

using std::cout;
using std::cerr;
using std::endl;

GLuint bufferObj;
cudaGraphicsResource* resource;

ProcGlGenBuffers glGenBuffers = nullptr;
ProcGlBindBuffer glBindBuffer = nullptr;
ProcGlBufferData glBufferData = nullptr;
ProcGlDeleteBuffers glDeleteBuffers = nullptr;

const int dim = 512;

__global__ void kernel(uchar4* ptr)
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

static void key_func(unsigned char key, int x, int y)
{
    cudaError error;

    switch (key)
    {
    case 27:
        error = cudaGraphicsUnregisterResource(resource);
        if (error != cudaSuccess)
            std::cerr << "cudaGraphicsUnregisterResource failed with error code " << error << endl;
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &bufferObj);
        exit(0);
        break;
    default:
        break;
    }
}

static void draw_func(void)
{
    glDrawPixels(dim, dim, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}

int main(int argc, char** argv)
{
    cudaDeviceProp prop;
    uchar4* devicePtr;
    size_t size;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;

    auto error = cudaChooseDevice(&dev, &prop);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaChooseDevice failed with error code " << error << endl;
        return 0;
    }

    cudaGLSetGLDevice(dev);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(dim, dim);
    glutCreateWindow("bitmap");

    glGenBuffers = (ProcGlGenBuffers)wglGetProcAddress("glGenBuffers");
    glBindBuffer = (ProcGlBindBuffer)wglGetProcAddress("glBindBuffer");
    glBufferData = (ProcGlBufferData)wglGetProcAddress("glBufferData");
    glDeleteBuffers = (ProcGlDeleteBuffers)wglGetProcAddress("glDeleteBuffers");

    glGenBuffers(1, &bufferObj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, 4 * dim * dim, nullptr, GL_DYNAMIC_DRAW_ARB);

    error = cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed with error code " << error << endl;
        return 0;
    }

    error = cudaGraphicsMapResources(1, &resource, nullptr);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaGraphicsMapResources failed with error code " << error << endl;
        return 0;
    }

    error = cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, resource);
    if (error != cudaSuccess)
    {
        std::cerr << "cudaGraphicsResourceGetMappedPointer failed with error code " << error << endl;
        return 0;
    }

    dim3 grids(dim / 16, dim / 16);
    dim3 threads(16, 16);

    kernel << <grids, threads >> > (devicePtr);

    cudaGraphicsUnmapResources(1, &resource, nullptr);

    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMainLoop();

    return 0;
}