#include "lstm_operations.h"
#include <iostream>
int main(void)
{
    Layer L = lstm_layer(2, 1);
    Vector state1 = lstm_state(1);
    Vector x(2);
    x << 1, 0.5;
    Vector state2 = lstm_state(1);

    state1 << 0, 0, 0, 0, 0.5, 0.75;
    lstm_forwardpass(L, state1, x, state2);

    Vector h = lstm_output(state2);
    Vector dh(1);
    dh << 0.67;

    std::cout << "h: " << h << "\n";
    std::cout << "dh: " << dh << "\n";
    // dh[0] = 0;

    Vector d = lstm_state(1);
    d << 0.1, 0.2, 0.3, 0.4, -0.1, -0.2;

    Matrix dW = Matrix::Zero(4 * 1, 2 + 1);
    Vector dx = Vector::Zero(2);

    lstm_backwardpass(L, state2, x, state1, dh, dW, d, dx);
}