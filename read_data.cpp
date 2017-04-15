#include <fstream>
#include <locale>
#include "read_data.h"

struct pipe_delimiter : std::ctype<char>
{
    pipe_delimiter() : std::ctype<char>(get_table()) {}
    static mask const *get_table()
    {
        static mask rc[table_size];
        rc['|'] = std::ctype_base::space;
        return &rc[0];
    }
};

std::vector<std::string *> read_data(const std::string &filename)
{
    std::vector<std::string *> v;
    std::ifstream fin(filename);
    fin.imbue(std::locale(fin.getloc(), new pipe_delimiter()));
    int n = 0;
    while (fin)
    {
        std::string *chunk = new std::string();
        fin >> *chunk;
        if (chunk->size())
            v.push_back(chunk);
    }
    return v;
}
