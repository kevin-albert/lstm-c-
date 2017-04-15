#ifndef optimizer_h
#define optimizer_h

namespace optimize
{

template <class W, class G, class M>
void nesterov(const double rate, const double momentum, W &w, const G &g, M &m)
{
    m = momentum * momentum * m - (1 + momentum) * rate * g;
    w = w + m;
}

template <class W, class G, class M, class... Next>
void nesterov(const double rate, const double momentum, W &w, const G &g, M &m, Next &... next)
{
    nesterov(rate, momentum, w, g, m);
    nesterov(rate, momentum, next...);
}
}

#endif