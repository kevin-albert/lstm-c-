#ifndef js_h
#define js_h

#include <string>
#include <fstream>
#include "core.h"

namespace js
{

class outfile
{
  private:
    std::ofstream out;

  public:
    outfile(const std::string &output_file) : out(output_file) {}

    outfile &operator<<(const Matrix &m);
    outfile &operator<<(const Vector &v);

    template <class T>
    outfile &operator<<(const T t);
};

outfile &outfile::operator<<(const Matrix &m)
{
    out << "new Matrix([";
    std::string row_sep = "";
    for (int row = 0; row < m.rows(); ++row)
    {
        out << row_sep << '[';
        if (m.cols())
        {
            out << m(row, 0);
            for (int col = 0; col < m.cols(); ++col)
                out << ',' << m(row, col);
        }
        out << ']';
        row_sep = ",\n";
    }
    out << "])";
    return *this;
}

outfile &outfile::operator<<(const Vector &v)
{
    out << "new Matrix([";
    std::string sep = "";
    for (int i = 0; i < v.size(); ++i)
    {
        out << sep << v[i];
        sep = ", ";
    }
    out << "])";
    return *this;
}

template <class T>
outfile &outfile::operator<<(const T t)
{
    out << t;
    return *this;
}
}

#endif