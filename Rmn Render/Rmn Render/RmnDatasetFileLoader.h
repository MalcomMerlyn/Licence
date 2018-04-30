#ifndef RMN_DATASET_FILE_LOADER

#include "cuda.h"
#include "cuda_runtime.h"

#include "ErrnoErrorMessage.h"

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

struct Color
{
    Color(unsigned int r, unsigned int g, unsigned int b, unsigned int a)
        : r{ r }, g{ g }, b{ b }, a{ a }
    { }

    ~Color() { }

    unsigned int r, g, b, a;
};

class RmnDatasetFileLoader
{
public:
    RmnDatasetFileLoader(string filePath, string fileName);

    void loadDataset();

    dim3 getRmnDatasetDimensions() const { return m_rmnDim; }
    dim3 getRmnSubsetPoint0() const { return m_subset0; }
    dim3 getRmnSubsetPoint1() const { return m_subset1; }
    const vector<unsigned int>& getColormap() const { return m_colormap; }
    const vector<Color>& getColor() const { return m_color; }

    const unsigned char* getRmnDataset() { return m_rmnDataset; }

    ~RmnDatasetFileLoader() {}

private:
    const string ConfigFileExtension = ".cfg";
    const string DataFileExtension = ".dat";

    void loadConfigurationData();
    void loadRmnDataset();

    string m_filePath;
    string m_fileName;
    string m_configFileName;
    string m_dataFileName;

    unsigned char* m_rmnDataset;

    dim3 m_rmnDim;
    dim3 m_subset0, m_subset1;
    vector<unsigned int> m_colormap;
    vector<Color> m_color;
};

#endif // !RMN_DATASET_FILE_LOADER
