#ifndef lstm_operations_h
#define lstm_operations_h

#include "lstm_core.h"

void lstm_forwardpass(const Layer &l,
                      const Vector &state1,
                      const Vector &x,
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

void lstm_backwardpass(const Layer &l,
                       const Vector &state2,
                       const Vector &x,
                       const Vector &state1,
                       const Vector &dh,
                       Matrix &dW,
                       Vector &d,
                       Vector &dx)
{
    const uint32_t nc = l.num_cells;
    const uint32_t ni = l.num_input;

    // d = [da; di; df; do; dc; dI(1+length(x):end)];

    // Compute gradients

    // Carry over BPTT computation from t+1
    // dc(t+1) is stored in d_next
    // dc = d_next(4 * num_cells + 1:5 * num_cells) + dh.*o.*(1 - tanh(c).^ 2);
    d.segment(4 * nc, nc).array() += dh.array() *
                                     state2.segment(3 * nc, nc).array() *
                                     (1 - (state2.segment(4 * nc, nc).array().tanh().pow(2)));

    // g'(x) = g(x)(1-g(x))
    // da = dc.*i.*(1 - a.^ 2);
    d.segment(0 * nc, nc) = d.segment(4 * nc, nc).array() *
                            state2.segment(1 * nc, nc).array() *
                            (1 - state2.segment(0 * nc, nc).array().pow(2));

    // di = dc.*a.*i.*(1 - i);
    d.segment(1 * nc, nc) = d.segment(4 * nc, nc).array() *
                            state2.segment(0 * nc, nc).array() *
                            state2.segment(1 * nc, nc).array() *
                            (1 - state2.segment(1 * nc, nc).array());

    // df = dc.*cp.*f.*(1 - f);
    d.segment(2 * nc, nc) = d.segment(4 * nc, nc).array() *
                            state1.segment(4 * nc, nc).array() *
                            state2.segment(2 * nc, nc).array() *
                            (1 - state2.segment(2 * nc, nc).array());

    // do = dh.*tanh(c).*o.*(1 - o);
    d.segment(3 * nc, nc) = dh.array() *
                            state2.segment(nc * 4, nc).array().tanh() *
                            state2.segment(3 * nc, nc).array() *
                            (1 - state2.segment(3 * nc, nc).array());

    // Now dc becomes dc(t-1)
    // dc = dc.*df;
    d.segment(4 * nc, nc).array() *= d.segment(2 * nc, nc).array();

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
    dW << d.segment(0 * nc, nc) * x.transpose(), d.segment(0 * nc, nc) * state1.segment(5 * nc, nc).transpose(),
        d.segment(1 * nc, nc) * x.transpose(), d.segment(1 * nc, nc) * state1.segment(5 * nc, nc).transpose(),
        d.segment(2 * nc, nc) * x.transpose(), d.segment(2 * nc, nc) * state1.segment(5 * nc, nc).transpose(),
        d.segment(3 * nc, nc) * x.transpose(), d.segment(3 * nc, nc) * state1.segment(5 * nc, nc).transpose();

    // dI = transpose(W) * dz
    // dx = dI(1:length(x))
    dx = l.W.block(0 * nc, 0, nc, ni).transpose() * d.segment(0 * nc, nc) +
         l.W.block(1 * nc, 0, nc, ni).transpose() * d.segment(1 * nc, nc) +
         l.W.block(2 * nc, 0, nc, ni).transpose() * d.segment(2 * nc, nc) +
         l.W.block(3 * nc, 0, nc, ni).transpose() * d.segment(3 * nc, nc);

    // dhp = dI(1+length(x):end)
    d.segment(5 * nc, nc) =
        l.W.block(0 * nc, ni, nc, nc).transpose() * d.segment(0 * nc, nc) +
        l.W.block(1 * nc, ni, nc, nc).transpose() * d.segment(1 * nc, nc) +
        l.W.block(2 * nc, ni, nc, nc).transpose() * d.segment(2 * nc, nc) +
        l.W.block(3 * nc, ni, nc, nc).transpose() * d.segment(3 * nc, nc);
}

#endif