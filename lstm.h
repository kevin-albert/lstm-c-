#ifndef lstm_h
#define lstm_h

#include <cmath>
#include "core.h"

struct Layer
{
    uint32_t num_input;
    uint32_t num_cells;
    Vector b;
    Matrix W;
};

Vector lstm_state(const uint32_t num_cells);

Layer lstm_layer(const uint32_t num_input, const uint32_t num_cells);
void lstm_init_layer(Layer &layer);

auto lstm_output(const Vector &state) -> decltype(state.segment(5 * state.size() / 6, state.size() / 6));

void lstm_forwardpass(const Layer &l,
                      const Vector &state1,
                      const Vector &x,
                      Vector &state2);

void lstm_backwardpass(const Layer &l,
                       const Vector &state2,
                       const Vector &x,
                       const Vector &state1,
                       const Vector &dh,
                       Matrix &dW,
                       Vector &d,
                       Vector &dx);
#endif