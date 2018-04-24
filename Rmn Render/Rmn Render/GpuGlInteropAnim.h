#ifndef GPU_GL_INTEROP_ANIM_H_
#define GPU_GL_INTEROP_ANIM_H_

#include <GL/glut.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "CudaErrorMessage.h"
#include "GLProcedures.h"

#include <iostream>
#include <stdexcept>
#include <string>

using std::runtime_error;
using std::string;
using std::to_string;
using std::cerr;
using std::endl;

class GpuGLAnim
{
public:
    long imageize(void) const { return m_height * m_width * sizeof(uchar4); }

    void setClickDrag(void(*fClickDrag)(void*, int, int, int, int))
    {
        m_fClickDrag = fClickDrag;
    }

    static void animAdExit(void(*fGenerateFrame)(uchar4*, void*, int), void(*fCleanup)(void*), int width, int height, void* dataBlock = nullptr)
    {
        static GpuGLAnim animBitmap(width, height, dataBlock);
        *getInstancePointer() = &animBitmap;

        animBitmap.m_fGenerateFrame = fGenerateFrame;
        animBitmap.m_fAnimExit = fCleanup;

        glutKeyboardFunc(s_key);
        glutDisplayFunc(s_draw);
        if (animBitmap.m_fClickDrag != nullptr)
            glutMouseFunc(s_mouse);
        glutIdleFunc(s_idle);
        glutMainLoop();
    }

    GpuGLAnim(GpuGLAnim const&) = delete;
    void operator=(GpuGLAnim const&) = delete;

private:
    GpuGLAnim(int width, int height, void* dataBlock = nullptr)
    {
        m_width = width;
        m_height = height;
        m_dataBlock = dataBlock;
        m_fClickDrag = nullptr;

        cudaDeviceProp prop;
        int dev;
        cudaError error;
        memset(&prop, 0, sizeof(cudaDeviceProp));
        prop.minor = 0;
        prop.major = 1;
        error = cudaChooseDevice(&dev, &prop);
        if (error != cudaSuccess)
            throw runtime_error(makeErrorMessage("cudaChooseDevice", error, __FILE__, __LINE__));

        int argc = 1;
        char* argv = "GpuGLAnim";
        glutInit(&argc, &argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowSize(width, height);
        glutCreateWindow("bitmap");

        glGenBuffers(1, &m_bufferObj);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_bufferObj);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW_ARB);

        error = cudaGraphicsGLRegisterBuffer(&m_resource, m_bufferObj, cudaGraphicsMapFlagsNone);
        if (error != cudaSuccess)
            throw runtime_error(makeErrorMessage("cudaGraphicsGLRegisterBuffer", error, __FILE__, __LINE__));
    }

    inline static GpuGLAnim** getInstancePointer()
    {
        static GpuGLAnim* instance;

        return &instance;
    }

    inline static GpuGLAnim* getInstance()
    {
        return *getInstancePointer();
    }

    ~GpuGLAnim()
    {
        cudaError error = cudaGraphicsUnregisterResource(m_resource);
        if (error != cudaSuccess)
            cerr << makeErrorMessage("cudaGraphicsUnregisterResource", error, __FILE__, __LINE__) << endl;

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &m_bufferObj);
    }

    static void s_mouse(int button, int state, int xPos, int yPos)
    {
        if (button == GLUT_LEFT_BUTTON)
        {
            if (state == GLUT_DOWN)
            {
                getInstance()->m_dragStartX = xPos;
                getInstance()->m_dragStartY = yPos;
            }
            else if (state == GLUT_UP)
            {
                getInstance()->m_fClickDrag(getInstance()->m_dataBlock, getInstance()->m_dragStartX, getInstance()->m_dragStartY, xPos, yPos);
            }
        }
    }

    static void s_idle(void)
    {
        static size_t m_ticks;

        cudaError error;
        uchar4* devPtr;
        size_t size;

        error = cudaGraphicsMapResources(1, &(getInstance()->m_resource), 0);
        if (error != cudaSuccess)
            throw runtime_error(makeErrorMessage("cudaGraphicsMapResources", error, __FILE__, __LINE__));

        error = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, getInstance()->m_resource);
        if (error != cudaSuccess)
            throw runtime_error(makeErrorMessage("cudaGraphicsResourceGetMappedPointer", error, __FILE__, __LINE__));

        getInstance()->m_fGenerateFrame(devPtr, getInstance()->m_dataBlock, m_ticks++);

        error = cudaGraphicsUnmapResources(1, &getInstance()->m_resource, 0);
        if (error != cudaSuccess)
            throw runtime_error(makeErrorMessage("cudaGraphicsUnmapResources", error, __FILE__, __LINE__));

        glutPostRedisplay();
    }

    static void s_key(unsigned char key, int x, int y)
    {
        switch (key)
        {
        case 27:
            //getInstance()->~GpuGLAnim();
            exit(0);
        }
    }

    static void s_draw(void)
    {
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(getInstance()->m_width, getInstance()->m_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glutSwapBuffers();
    }

    GLuint m_bufferObj;
    cudaGraphicsResource* m_resource;
    int m_width, m_height;
    void* m_dataBlock;
    void(*m_fGenerateFrame)(uchar4*, void*, int);
    void(*m_fAnimExit)(void*);
    void(*m_fClickDrag)(void*, int, int, int, int);
    int m_dragStartX, m_dragStartY;
};

#endif // !GPU_GL_INTEROP_ANIM_H_

