#ifndef read_file_h
#define read_file_h

#include <vector>
#include <string>

std::vector<std::string *> read_data(const std::string &filename);
bool file_exists(const std::string &filename);

#endif