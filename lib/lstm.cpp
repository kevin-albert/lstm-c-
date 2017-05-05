#include <limits>
#include "lstm.h"
#include "rng.h"

Vector Sentinel::_no_arg = Vector::Constant(999999, std::numeric_limits<float>::quiet_NaN());

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

    layer.W = Matrix::Zero(4 * num_cells, num_input + num_cells);

    //
    // Biases look like:
    // [ ba  bi  bf  bo ]
    // bf starts out at 5
    //
    layer.b = Vector::Zero(4 * num_cells);

    return layer;
}

void lstm_init_layer(Layer &layer)
{
    rng::setnormal(layer.W, 0, 1.0 / std::sqrt(layer.num_input + layer.num_cells));
    layer.b.setRandom();
    layer.b *= 0.1;
    layer.b.segment(2 * layer.num_cells, layer.num_cells) = Vector::Constant(layer.num_cells, 5);
}

Gradients lstm_gradients(const uint32_t num_input, const uint32_t num_cells)
{
    Gradients g;
    g.W = Matrix::Zero(4 * num_cells, num_input + num_cells);
    g.S = lstm_state(num_cells);
    return g;
}

auto lstm_output(const Vector &state) -> decltype(state.segment(5 * state.size() / 6, state.size() / 6))
{
    return state.segment(5 * state.size() / 6, state.size() / 6);
}
