#include "lstm_operations.h"
#include <iostream>
int main(void)
{
    std::cout << "n=" << Eigen::nbThreads() << "\n";
    int ni = 100;
    int nc = 2000;
    Layer L = lstm_layer(ni, nc);
    Vector state = lstm_state(nc);
    Vector x = Vector::Random(ni);
    Vector dx = Vector::Zero(ni);
    Vector dh = Vector::Random(nc);
    Vector d = lstm_state(nc);

    Matrix dW = Matrix::Zero(4 * nc, ni + nc);

    for (int i = 0; i < 100; ++i)
    {
        lstm_backwardpass(L, state, x, state, dh, dW, d, dx);
    }
}