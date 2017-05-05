#include <fstream>
#include <sstream>
#include "read_data.h"

#include <iostream>
std::vector<std::string *> read_data(const std::string &filename)
{
    std::vector<std::string *> v;
    std::string *str = new std::string();
    v.push_back(str);
    std::ifstream fin(filename);
    std::stringstream buffer;
    buffer << fin.rdbuf();
    *str = buffer.str();
    return v;
}
