#ifndef GL_PROCEDURES_H_
#define GL_PROCEDURES_H_

#include <GL/glut.h>

#include "cuda_gl_interop.h"

typedef void(*ProcGlGenBuffers)(GLsizei n, GLuint* buffers);
typedef void(*ProcGlBindBuffer)(GLenum target, GLuint buffer);
typedef void(*ProcGlBufferData)(GLenum target, GLsizei size, const GLvoid* data, GLenum usage);
typedef void(*ProcGlDeleteBuffers)(GLsizei n, const GLuint* buffers);

extern ProcGlGenBuffers glGenBuffers;
extern ProcGlBindBuffer glBindBuffer;
extern ProcGlBufferData glBufferData;
extern ProcGlDeleteBuffers glDeleteBuffers;

static void LoadAndExecuteProcGlGenBuffers(GLsizei n, GLuint* buffers)
{
    glGenBuffers = (ProcGlGenBuffers)wglGetProcAddress("glGenBuffers");

    glGenBuffers(n, buffers);
}

static void LoadAndExecuteProcGlBindBuffer(GLenum target, GLuint buffer)
{
    glBindBuffer = (ProcGlBindBuffer)wglGetProcAddress("glBindBuffer");

    glBindBuffer(target, buffer);
}

static void LoadAndExecuteProcGlBufferData(GLenum target, GLsizei size, const GLvoid* data, GLenum usage)
{
    glBufferData = (ProcGlBufferData)wglGetProcAddress("glBufferData");

    glBufferData(target, size, data, usage);
}

static void LoadAndExecuteProcGlDeleteBuffers(GLsizei n, const GLuint* buffers)
{
    glDeleteBuffers = (ProcGlDeleteBuffers)wglGetProcAddress("glDeleteBuffers");

    glDeleteBuffers(n, buffers);
}

ProcGlGenBuffers glGenBuffers = LoadAndExecuteProcGlGenBuffers;
ProcGlBindBuffer glBindBuffer = LoadAndExecuteProcGlBindBuffer;
ProcGlBufferData glBufferData = LoadAndExecuteProcGlBufferData;
ProcGlDeleteBuffers glDeleteBuffers = LoadAndExecuteProcGlDeleteBuffers;

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

#endif // !GL_PROCEDURES_H_
