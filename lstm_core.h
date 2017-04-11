#ifndef lstm_core_h
#define lstm_core_h

#include <cmath>
#include "Eigen/Dense"

typedef Eigen::VectorXf Vector;
typedef Eigen::MatrixXf Matrix;

struct Layer
{
    uint32_t num_input;
    uint32_t num_cells;
    Vector b;
    Matrix W;
};

#include <iostream>
Vector lstm_state(const uint32_t num_cells)
{
    return Vector::Zero(6 * num_cells);
}

Layer lstm_layer(const uint32_t num_input, const uint32_t num_cells)
{
    Layer layer;
    layer.num_input = num_input;
    layer.num_cells = num_cells;

    //
    // Weights look like:
    // +----------+
    // | Wax  Wah |
    // | Wix  Wih |
    // | Wfx  Wfh |
    // | Wox  Woh |
    // +----------+
    //

    layer.W = Matrix::Random(4 * num_cells, num_input + num_cells) / std::sqrt(num_input + num_cells);

    //
    // Biases look like:
    // [ ba  bi  bf  bo ]
    // bf starts out at 5
    //
    layer.b = Vector::Random(4 * num_cells);
    layer.b.segment(2 * num_cells, num_cells) = Vector::Constant(num_cells, 5);

    return layer;
}

auto lstm_output(const Vector &state) -> decltype(state.segment(5 * state.size() / 6, state.size() / 6))
{
    return state.segment(5 * state.size() / 6, state.size() / 6);
}

void tanh(Vector &v)
{
    for (uint32_t i = 0; i < v.size(); ++i)
        v.data()[i] = std::tanhf(v.data()[i]);
}

void lsig(Vector &v)
{
    for (uint32_t i = 0; i < v.size(); ++i)
        v.data()[i] = (1.0F + tanhf(v.data()[i] / 2)) / 2;
}

#endif