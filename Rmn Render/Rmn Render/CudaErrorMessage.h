#ifndef CUDA_ERROR_MESSAGE_H_
#define CUDA_ERROR_MESSAGE_H_

#include "cuda_runtime.h"

#include <string>

using std::string;
using std::to_string;

static string makeErrorMessage(cudaError error)
{
    string errorName(cudaGetErrorName(error));
    string errorString(cudaGetErrorString(error));

    return "Cuda failure : error code " + errorName + " : " + errorString;
}

static string makeErrorMessage(cudaError error, const char* file, int line)
{
    string errorPlace;
    string errorName(cudaGetErrorName(error));
    string errorString(cudaGetErrorString(error));

    errorPlace = "Error occured in file ";
    errorPlace += file;
    errorPlace += " at line ";
    errorPlace += to_string(line);

    return errorPlace + " : cuda failure : error code " + errorName + " : " + errorString;
}

static string makeErrorMessage(string failedFunction, cudaError error)
{
    string errorName(cudaGetErrorName(error));
    string errorString(cudaGetErrorString(error));

    return "Function " + failedFunction + " failed with error code " + errorName + " : " + errorString;
}

static string makeErrorMessage(string failedFunction, cudaError error, const char* file, int line)
{
    string errorPlace;
    string errorName(cudaGetErrorName(error));
    string errorString(cudaGetErrorString(error));

    errorPlace = "Error occured in file ";
    errorPlace += file;
    errorPlace += " at line ";
    errorPlace += to_string(line);

    return errorPlace + " : function " + failedFunction + " failed with error code " + errorName + " : " + errorString;
}

#endif // !CUDA_ERROR_MESSAGE_H_
