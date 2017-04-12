#ifndef checkpoint_h
#define checkpoint_h

#include <iostream>
#include <fstream>
#include "lstm_core.h"

// API
template <class... Vars>
void checkpoint_save(const std::string &filename, const Vars &... vars);

template <class... Vars>
void checkpoint_load(const std::string &filename, Vars &... vars);

//
// Save
//
template <class T>
void checkpoint_save_scalar(std::ostream &out, const T x) { out.write((char *)&x, sizeof x); }
void checkpoint_save_var(std::ostream &out, const int i) { checkpoint_save_scalar(out, i); }
void checkpoint_save_var(std::ostream &out, const long int i) { checkpoint_save_scalar(out, i); }
void checkpoint_save_var(std::ostream &out, const uint32_t i) { checkpoint_save_scalar(out, i); }
void checkpoint_save_var(std::ostream &out, const uint64_t i) { checkpoint_save_scalar(out, i); }
void checkpoint_save_var(std::ostream &out, const float f) { checkpoint_save_scalar(out, f); }
void checkpoint_save_var(std::ostream &out, const double f) { checkpoint_save_scalar(out, f); }

void checkpoint_save_var(std::ostream &out, const Matrix &m)
{
    checkpoint_save_scalar(out, m.rows());
    checkpoint_save_scalar(out, m.cols());
    out.write((char *)m.data(), m.size() * sizeof(float));
}

template <class Var>
void checkpoint_save(std::ostream &out, const Var &v)
{
    checkpoint_save_var(out, v);
}

template <class Var, class... Vars>
void checkpoint_save(std::ostream &out, const Var &v, const Vars &... vars)
{
    checkpoint_save(out, v);
    checkpoint_save(out, vars...);
}

template <class... Vars>
void checkpoint_save(const std::string &filename, const Vars &... vars)
{
    std::ofstream fout(filename);
    checkpoint_save(fout, vars...);
}

//
// Load
//
template <class T>
void checkpoint_load_scalar(std::istream &in, T &x) { in.read((char *)&x, sizeof x); }
void checkpoint_load_var(std::istream &in, int &i) { checkpoint_load_scalar(in, i); }
void checkpoint_load_var(std::istream &in, long int &i) { checkpoint_load_scalar(in, i); }
void checkpoint_load_var(std::istream &in, uint32_t &i) { checkpoint_load_scalar(in, i); }
void checkpoint_load_var(std::istream &in, uint64_t &i) { checkpoint_load_scalar(in, i); }
void checkpoint_load_var(std::istream &in, float &f) { checkpoint_load_scalar(in, f); }
void checkpoint_load_var(std::istream &in, double &f) { checkpoint_load_scalar(in, f); }

// just treat it like a matrix?
template <class M>
void checkpoint_load_var(std::istream &in, M &m)
{
    decltype(m.rows()) rows;
    decltype(m.cols()) cols;
    checkpoint_load_var(in, rows);
    checkpoint_load_var(in, cols);
    m.resize(rows, cols);
    in.read((char *)m.data(), m.size() * sizeof(float));
}

template <class Var>
void checkpoint_load(std::istream &in, Var &v)
{
    checkpoint_load_var(in, v);
}

template <class Var, class... Vars>
void checkpoint_load(std::istream &in, Var &v, Vars &... vars)
{
    checkpoint_load(in, v);
    checkpoint_load(in, vars...);
}

template <class... Vars>
void checkpoint_load(const std::string &filename, Vars &... vars)
{
    std::ifstream fin(filename);
    checkpoint_load(fin, vars...);
}

#endif