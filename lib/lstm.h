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

struct Gradients
{
    Matrix W;
    Vector S;
};

struct Sentinel
{
    static Vector _no_arg;
    static bool has_arg(const Vector &v) { return v.size() != _no_arg.size(); }
};

Vector lstm_state(const uint32_t num_cells);
Layer lstm_layer(const uint32_t num_input, const uint32_t num_cells);
void lstm_init_layer(Layer &layer);
Gradients lstm_gradients(const uint32_t num_input, const uint32_t num_cells);

auto lstm_output(const Vector &state) -> decltype(state.segment(5 * state.size() / 6, state.size() / 6));

template <class V>
void softmax_activation(const V &y, Vector &p);

template <class V>
float softmax_cross_entropy_onehot(const Vector &y_, const V &y, Vector &p, Vector &dy);

template <class X>
void lstm_forwardpass(const Layer &l,
                      const Vector &state1,
                      const X &x,
                      Vector &state2)
{
    uint32_t ni = l.num_input;
    uint32_t nc = l.num_cells;

    // Calc all gates, input values
    //z = W*[x;h] + b;
    state2.segment(0, 4 * nc) = l.W.block(0, 0, 4 * nc, ni) * x +
                                l.W.block(0, ni, 4 * nc, nc) * state1.segment(5 * nc, nc) +
                                l.b;

    // a = tanh(z(1:num_cells));
    state2.segment(0 * nc, nc) = state2.segment(0 * nc, nc).array().tanh();

    // i = lsig(z(1 + num_cells : 2 * num_cells));
    state2.segment(1 * nc, nc) = (1 + ((state2.segment(1 * nc, nc) / 2).array().tanh())) / 2;

    // f = lsig(z(1 + 2 * num_cells : 3 * num_cells));
    state2.segment(2 * nc, nc) = (1 + ((state2.segment(2 * nc, nc) / 2).array().tanh())) / 2;

    // o = lsig(z(1 + 3 * num_cells : 4 * num_cells));
    state2.segment(3 * nc, nc) = (1 + ((state2.segment(3 * nc, nc) / 2).array().tanh())) / 2;

    // % Calc cell state
    // % note - cell state is linear (not tanh'd)
    // c = a .* i + cp .* f;
    state2.segment(4 * nc, nc) = state2.segment(0 * nc, nc).array() * state2.segment(1 * nc, nc).array() +
                                 state1.segment(4 * nc, nc).array() * state2.segment(2 * nc, nc).array();

    // h = tanh(c) .* o;
    state2.segment(5 * nc, nc) = state2.segment(4 * nc, nc).array().tanh() * state2.segment(3 * nc, nc).array();
}

template <class X>
void lstm_backwardpass(bool has_next,
                       const Layer &l,
                       const Vector &state2,
                       const X &x,
                       const Vector &state1,
                       const Vector &dh,
                       Gradients &g,
                       Vector &dx = Sentinel::_no_arg)
{
    const uint32_t nc = l.num_cells;
    const uint32_t ni = l.num_input;

    // d = [da; di; df; do; dc; dI(1+length(x):end)];

    // Compute gradients

    // Carry over BPTT computation from t+1
    // dc(t+1) is stored in d_next
    // dc = d_next(4 * num_cells + 1:5 * num_cells) + dh.*o.*(1 - tanh(c).^ 2);
    if (has_next)
    {
        g.S.segment(4 * nc, nc).array() += dh.array() *
                                           state2.segment(3 * nc, nc).array() *
                                           (1 - (state2.segment(4 * nc, nc).array().tanh().pow(2)));
    }

    // g'(x) = g(x)(1-g(x))
    // da = dc.*i.*(1 - a.^ 2);
    g.S.segment(0 * nc, nc) = g.S.segment(4 * nc, nc).array() *
                              state2.segment(1 * nc, nc).array() *
                              (1 - state2.segment(0 * nc, nc).array().pow(2));

    // di = dc.*a.*i.*(1 - i);
    g.S.segment(1 * nc, nc) = g.S.segment(4 * nc, nc).array() *
                              state2.segment(0 * nc, nc).array() *
                              state2.segment(1 * nc, nc).array() *
                              (1 - state2.segment(1 * nc, nc).array());

    // df = dc.*cp.*f.*(1 - f);
    g.S.segment(2 * nc, nc) = g.S.segment(4 * nc, nc).array() *
                              state1.segment(4 * nc, nc).array() *
                              state2.segment(2 * nc, nc).array() *
                              (1 - state2.segment(2 * nc, nc).array());

    // do = dh.*tanh(c).*o.*(1 - o);
    g.S.segment(3 * nc, nc) = dh.array() *
                              state2.segment(nc * 4, nc).array().tanh() *
                              state2.segment(3 * nc, nc).array() *
                              (1 - state2.segment(3 * nc, nc).array());

    // Now dc becomes dc(t-1)
    // dc = dc.*df;
    g.S.segment(4 * nc, nc).array() *= g.S.segment(2 * nc, nc).array();

    // // Weights, dx, etc
    // dz = [da; di; df; do];
    // I = [x; hp];

    // dW = dz * transpose(I);
    // dz * IT
    //  = [ da; di; df; do ] * [ transpose(x), transpose(hp) ];
    //  = [ da * transpose(x), da * transpose(hp);
    //      di * transpose(x), di * transpose(hp);
    //      df * transpose(x), df * transpose(hp);
    //      do * transpose(x), do * transpose(hp);
    g.W << g.S.segment(0 * nc, nc) * x.transpose(), g.S.segment(0 * nc, nc) * state1.segment(5 * nc, nc).transpose(),
        g.S.segment(1 * nc, nc) * x.transpose(), g.S.segment(1 * nc, nc) * state1.segment(5 * nc, nc).transpose(),
        g.S.segment(2 * nc, nc) * x.transpose(), g.S.segment(2 * nc, nc) * state1.segment(5 * nc, nc).transpose(),
        g.S.segment(3 * nc, nc) * x.transpose(), g.S.segment(3 * nc, nc) * state1.segment(5 * nc, nc).transpose();

    if (Sentinel::has_arg(dx))
    {
        // dI = transpose(W) * dz
        // dx = dI(1:length(x))
        dx = l.W.block(0 * nc, 0, nc, ni).transpose() * g.S.segment(0 * nc, nc) +
             l.W.block(1 * nc, 0, nc, ni).transpose() * g.S.segment(1 * nc, nc) +
             l.W.block(2 * nc, 0, nc, ni).transpose() * g.S.segment(2 * nc, nc) +
             l.W.block(3 * nc, 0, nc, ni).transpose() * g.S.segment(3 * nc, nc);
    }

    // dhp = dI(1+length(x):end)
    g.S.segment(5 * nc, nc) =
        l.W.block(0 * nc, ni, nc, nc).transpose() * g.S.segment(0 * nc, nc) +
        l.W.block(1 * nc, ni, nc, nc).transpose() * g.S.segment(1 * nc, nc) +
        l.W.block(2 * nc, ni, nc, nc).transpose() * g.S.segment(2 * nc, nc) +
        l.W.block(3 * nc, ni, nc, nc).transpose() * g.S.segment(3 * nc, nc);
}

template <class V>
void softmax_activation(const V &y, Vector &p)
{
    p = y.array().exp();
    p = p / p.sum();
}

template <class V>
float softmax_cross_entropy_onehot(const Vector &y_, const V &y, Vector &p, Vector &dy)
{
    softmax_activation(y, p);

    // find the "hot" digit
    float max = y_[0];
    uint32_t k = 0;
    for (uint32_t i = 0; i < y_.size(); ++i)
        if (y_[i] > max)
        {
            max = y_[i];
            k = i;
        }

    dy = p - y_;
    if (p[k] <= 0)
        p[k] = 0.00001;

    return -std::log(p[k]);
}
#endif