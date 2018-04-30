#ifndef ERRNO_ERROR_MESSAGE_H_
#define ERRNO_ERROR_MESSAGE_H_

#include <string.h>
#include <errno.h>

#include <string>

using std::string;
using std::to_string;

const size_t BufferLength = 1000;

static string makeErrnoErrorMessage()
{
    char errorString[BufferLength];
    errno_t error = errno;

    strerror_s(errorString, BufferLength, errno);

    return "System failure : error code " + to_string(error) + " : " + errorString;
}

static string makeErrnoErrorMessage(const char* file, int line)
{
    string errorPlace;
    char errorString[BufferLength];
    errno_t error = errno;

    strerror_s(errorString, BufferLength, errno);

    errorPlace = "Error occured in file ";
    errorPlace += file;
    errorPlace += " at line ";
    errorPlace += to_string(line);

    return errorPlace + " : system failure : error code " + to_string(error) + " : " + errorString;
}

static string makeErrnoErrorMessage(string failedFunction)
{
    string errorPlace;
    char errorString[BufferLength];
    errno_t error = errno;

    strerror_s(errorString, BufferLength, errno);

    return "Function " + failedFunction + " failed with error code " + to_string(error) + " : " + errorString;
}

static string makeErrnoErrorMessage(string failedFunction, const char* file, int line)
{
    string errorPlace;
    char errorString[BufferLength];
    errno_t error = errno;

    strerror_s(errorString, BufferLength, errno);

    errorPlace = "Error occured in file ";
    errorPlace += file;
    errorPlace += " at line ";
    errorPlace += to_string(line);

    return errorPlace + " : function " + failedFunction + " failed with error code " + to_string(error) + " : " + errorString;
}

#endif // !ERRNO_ERROR_MESSAGE_H_
