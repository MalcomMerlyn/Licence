#include "RmnDatasetFileLoader.h"

RmnDatasetFileLoader::RmnDatasetFileLoader(string filePath, string fileName)
    : m_filePath{ filePath }
    , m_fileName{ fileName }
{
    m_configFileName = filePath + "/" + fileName + ConfigFileExtension;
    m_dataFileName = filePath + "/" + fileName + DataFileExtension;
}

void RmnDatasetFileLoader::loadDataset()
{
    loadConfigurationData();
    loadRmnDataset();
}

void RmnDatasetFileLoader::loadConfigurationData()
{
    ifstream configFile;
    string line;

    configFile.open(m_configFileName);

    for (getline(configFile, line); line.length() != 0; getline(configFile, line))
    {
        string value = line.substr(line.find(": ") + 2, line.length());
        istringstream stream(value);

        if (line.find("data_size.x") != string::npos)
            stream >> m_rmnDim.x;

        if (line.find("data_size.y") != string::npos)
            stream >> m_rmnDim.y;

        if (line.find("data_size.z") != string::npos)
            stream >> m_rmnDim.z;

        if (line.find("subset0.x") != string::npos)
            stream >> m_subset0.x;

        if (line.find("subset0.y") != string::npos)
            stream >> m_subset0.y;

        if (line.find("subset0.z") != string::npos)
            stream >> m_subset0.z;

        if (line.find("subset1.x") != string::npos)
            stream >> m_subset1.x;

        if (line.find("subset1.y") != string::npos)
            stream >> m_subset1.y;

        if (line.find("subset1.z") != string::npos)
            stream >> m_subset1.z;

        if (line.find("rgba") != string::npos)
        {
            float r, g, b, a;

            stream >> r >> g >> b >> a;

            m_color.emplace_back(r, g, b, a);
        }

        if (line.find("colormap") != string::npos)
        {
            vector<string> colors(istream_iterator<string>{ stream }, istream_iterator<string>());

            for (auto color : colors)
                m_colormap.push_back(stoi(color));
        }
    }

    configFile.close();
}

void RmnDatasetFileLoader::loadRmnDataset()
{
    FILE* f;
    int actual, total, expected, x, y;
    long offset;
    dim3 datasetDim;

    datasetDim.x = m_rmnDim.x;
    datasetDim.y = m_rmnDim.y;
    datasetDim.z = m_rmnDim.z;

    m_rmnDim.x = m_subset1.x - m_subset0.x;
    m_rmnDim.y = m_subset1.y - m_subset0.y;
    m_rmnDim.z = m_subset1.z - m_subset0.z;

    m_rmnDataset = (unsigned char*)malloc(m_rmnDim.x * m_rmnDim.y * m_rmnDim.z * sizeof(unsigned char));
    if (m_rmnDataset == nullptr)
        throw runtime_error(makeErrnoErrorMessage("malloc", __FILE__, __LINE__));

    f = fopen(m_dataFileName.c_str(), "rb");
    if (f == nullptr)
        throw runtime_error(makeErrnoErrorMessage("fopen", __FILE__, __LINE__));

    total = 0;
    for (x = m_subset0.x; x < m_subset1.x; x++)
        for (y = m_subset0.y; y < m_subset1.y; y++)
        {
            offset = x * datasetDim.y * datasetDim.z + y * datasetDim.z + m_subset0.z;

            auto error = fseek(f, offset, SEEK_SET);
            if (error != 0)
                throw runtime_error(makeErrnoErrorMessage("fseek", __FILE__, __LINE__));

            expected = m_rmnDim.z;
            actual = 0;
            while (actual < expected) 
            {
                auto readBytes = fread(m_rmnDataset + total + actual, 1, expected - actual, f);
                if (readBytes == 0)
                    throw runtime_error(makeErrnoErrorMessage("fread", __FILE__, __LINE__));

                actual += readBytes;
            }
            total += actual;
        }

    fclose(f);
}
