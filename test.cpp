#include "lstm_operations.h"
#include "lstm_core.h"
#include "checkpoint.h"

#include <iostream>
#include <fstream>
#include <unistd.h>

int main(int argc, char **argv)
{

    Matrix A = Matrix::Random(5, 2);
    // Vector v = Vector::Random(7);
    float v = 0.001;
    Matrix B = Matrix::Random(7, 7);

    checkpoint_save("checkpoint.dat", A, v, B);

    Matrix A_;
    float v_;
    Matrix B_;

    checkpoint_load("checkpoint.dat", A_, v_, B_);

    std::cout << "before:\n"
              << "A:\n"
              << A << "\n"
              << "v:\n"
              << v << "\n"
              << "B:\n"
              << B << "\n\n"
              << "after:\n"
              << "A:\n"
              << A_ << "\n"
              << "v:\n"
              << v_ << "\n"
              << "B:\n"
              << B_ << "\n";

    // Layer L = lstm_layer(2, 1);
    // Vector state1 = lstm_state(1);
    // Vector o = lstm_output(state1);
    // return o[0];

    // Vector x(2);
    // x << 1, 0.5;
    // Vector state2 = lstm_state(1);

    // state1 << 0, 0, 0, 0, 0.5, 0.75;
    // lstm_forwardpass(L, state1, x, state2);

    // Vector h = lstm_output(state2);
    // Vector dh(1);
    // dh << 0.67;

    // std::cout << "h: " << h << "\n";
    // std::cout << "dh: " << dh << "\n";
    // // dh[0] = 0;

    // Vector d = lstm_state(1);
    // d << 0.1, 0.2, 0.3, 0.4, -0.1, -0.2;

    // Matrix dW = Matrix::Zero(4 * 1, 2 + 1);
    // Vector dx = Vector::Zero(2);

    // lstm_backwardpass(L, state2, x, state1, dh, dW, d, dx);
}