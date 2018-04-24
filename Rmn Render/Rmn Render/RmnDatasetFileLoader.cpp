#include "RmnDatasetFileLoader.h"

RmnDatasetFileLoader::RmnDatasetFileLoader(string filePath, string fileName)
    : m_filePath{ filePath }
    , m_fileName{ fileName }
{
    string configFileName = filePath + fileName + ConfigFileExtension;
    string dataFileName = filePath + fileName + DataFileExtension;

    ifstream configFile;
    string line;

    configFile.open(configFileName);

    for (getline(configFile, line); line.length() != 0; getline(configFile, line))
    {
        if (line.find("data_size.x") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            stream >> m_rmnDim.x;
        }

        if (line.find("data_size.y") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            stream >> m_rmnDim.y;
        }

        if (line.find("data_size.z") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            stream >> m_rmnDim.z;
        }

        if (line.find("subset0.x") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            stream >> m_subset0.x;
        }

        if (line.find("subset0.y") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            stream >> m_subset0.y;
        }

        if (line.find("subset0.z") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            stream >> m_subset0.z;
        }

        if (line.find("subset1.x") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            stream >> m_subset1.x;
        }

        if (line.find("subset1.y") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            stream >> m_subset1.y;
        }

        if (line.find("subset1.z") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            stream >> m_subset1.z;
        }

        if (line.find("rgba") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);
            float r, g, b, a;

            stream >> r >> g >> b >> a;

            m_color.emplace_back(r, g, b, a);
        }

        if (line.find("colormap") != string::npos)
        {
            string value = line.substr(line.find(": ") + 2, line.length());
            istringstream stream(value);

            vector<string> colors(istream_iterator<string>{ stream }, istream_iterator<string>());

            for (auto color : colors)
                m_colormap.push_back(stoi(color));
        }
    }
}
