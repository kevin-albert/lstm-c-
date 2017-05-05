#ifndef checkpoint_h
#define checkpoint_h

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include "core.h"

namespace checkpoint
{
// API
template <class... Vars>
void save(const std::string &filename, const Vars &... vars);

template <class... Vars>
void load(const std::string &filename, Vars &... vars);

//
// Save
//
template <class T>
void save_scalar(std::ostream &out, const T x)
{
    out.write((char *)&x, sizeof x);
    if (!out)
    {
        std::cerr << "write failed\n";
        exit(EXIT_FAILURE);
    }
}
void save_var(std::ostream &out, const int i) { save_scalar(out, i); }
void save_var(std::ostream &out, const long int i) { save_scalar(out, i); }
void save_var(std::ostream &out, const uint32_t i) { save_scalar(out, i); }
void save_var(std::ostream &out, const uint64_t i) { save_scalar(out, i); }
void save_var(std::ostream &out, const size_t i) { save_scalar(out, i); }
void save_var(std::ostream &out, const float f) { save_scalar(out, f); }
void save_var(std::ostream &out, const double f) { save_scalar(out, f); }

void save_var(std::ostream &out, const Matrix &m)
{
    save_scalar(out, m.rows());
    save_scalar(out, m.cols());
    out.write((char *)m.data(), m.size() * sizeof(float));
    if (!out)
    {
        std::cerr << "write failed\n";
        exit(EXIT_FAILURE);
    }
}

template <class T>
void save_var(std::ostream &out, const std::vector<T> &v)
{
    save_scalar(out, (size_t)v.size());
    std::for_each(v.begin(), v.end(), [&out](const T x) { save_var(out, x); });
}

template <class Var>
void save(std::ostream &out, const Var &v)
{
    save_var(out, v);
}

template <class Var, class... Vars>
void save(std::ostream &out, const Var &v, const Vars &... vars)
{
    save(out, v);
    save(out, vars...);
}

template <class... Vars>
void save(const std::string &filename, const Vars &... vars)
{
    std::string filename_tmp = filename + ".tmp";
    std::ofstream fout(filename_tmp);
    save(fout, vars...);
    fout.close();
    std::rename(filename_tmp.c_str(), filename.c_str());
}

//
// Load
//
template <class T>
void load_scalar(std::istream &in, T &x)
{
    in.read((char *)&x, sizeof x);
    if (!in)
    {
        std::cerr << "read failed\n";
        exit(EXIT_FAILURE);
    }
}
void load_var(std::istream &in, int &i) { load_scalar(in, i); }
void load_var(std::istream &in, long int &i) { load_scalar(in, i); }
void load_var(std::istream &in, uint32_t &i) { load_scalar(in, i); }
void load_var(std::istream &in, uint64_t &i) { load_scalar(in, i); }
void load_var(std::istream &in, size_t &i) { load_scalar(in, i); }
void load_var(std::istream &in, float &f) { load_scalar(in, f); }
void load_var(std::istream &in, double &f) { load_scalar(in, f); }

template <class T>
void load_var(std::istream &in, std::vector<T> &v)
{
    size_t size;
    load_var(in, size);
    v.resize(size);
    for (size_t i = 0; i < size; ++i)
        load_var(in, v[i]);
}

// just treat it like a matrix?
template <class M>
void load_var(std::istream &in, M &m)
{
    decltype(m.rows()) rows;
    decltype(m.cols()) cols;
    load_var(in, rows);
    load_var(in, cols);
    m.resize(rows, cols);
    in.read((char *)m.data(), m.size() * sizeof(float));
    if (!in)
    {
        std::cerr << "read failed\n";
        exit(EXIT_FAILURE);
    }
}

template <class Var>
void load(std::istream &in, Var &v)
{
    load_var(in, v);
}

template <class Var, class... Vars>
void load(std::istream &in, Var &v, Vars &... vars)
{
    load(in, v);
    load(in, vars...);
}

template <class... Vars>
void load(const std::string &filename, Vars &... vars)
{
    std::ifstream fin(filename);
    load(fin, vars...);
}
}
#endif