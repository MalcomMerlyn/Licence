#ifndef RMN_DATASET_FILE_LOADER

#include "cuda.h"
#include "cuda_runtime.h"

#include <exception>
#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using std::getline;
using std::exception;
using std::ifstream;
using std::istringstream;
using std::istream_iterator;
using std::string;
using std::ios;
using std::runtime_error;
using std::vector;

class RmnDatasetFileLoader
{
public:
    RmnDatasetFileLoader(string filePath, string fileName);

    dim3 getRmnDatasetDimensions() { return m_rmnDim; }
    dim3 getRmnSubsetPoint0() { return m_subset0; }
    dim3 getRmnSubsetPoint1() { return m_subset1; }

    ~RmnDatasetFileLoader() {}

private:
    const string ConfigFileExtension = ".cfg";
    const string DataFileExtension = ".dat";

    string m_filePath;
    string m_fileName;

    dim3 m_rmnDim;
    dim3 m_subset0, m_subset1;
    vector<unsigned int> m_colormap;
    vector<uint4> m_color;
};

#endif // !RMN_DATASET_FILE_LOADER
