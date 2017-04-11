#include "lstm_operations.h"
#include <iostream>
int main(void)
{
    std::cout << "n=" << Eigen::nbThreads() << "\n";
    Layer L = lstm_layer(80, 500);
    Vector state = lstm_state(500);
    Vector x = Vector::Random(80);
    Vector dx = Vector::Zero(80);
    Vector dh = Vector::Random(500);
    Vector d = lstm_state(500);

    Matrix dW = Matrix::Zero(4 * 500, 80 + 500);

    for (int i = 0; i < 5000; ++i)
    {
        lstm_backwardpass(L, state, x, state, dh, dW, d, dx);
    }
}