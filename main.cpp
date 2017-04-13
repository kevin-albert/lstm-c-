#include <iostream>
#include <fstream>
#include <unistd.h>

#include "core.h"
#include "lstm.h"
#include "checkpoint.h"
#include "read_data.h"
#include "text_mapper.h"
#include "rng.h"

template <class T>
void debug(const std::string &name, const std::vector<T> &v)
{
    std::cout << name << ": ";
    for (int i = 0; i < v.size(); ++i)
        std::cout << v[i] << " ";
    std::cout << "\n";
}

template <class V>
void debug(const std::string &name, const V &v)
{
    std::cout << name << ": " << v << "\n";
}

#define dbg(var) debug(#var, var);

void init(int epochs,                      // -e
          int seq_length,                  // -s
          double rate,                     // -r
          double momentum,                 // -m
          double rate_decay,               // -d
          int num_cells,                   // -c
          const std::string &data_file,    // -i
          const std::string &output_file); // -o

void train(const std::string &data_file, const std::string &checkpoint_file);

void sample(const std::string &data_file, const std::string &checkpoint_file);

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        std::cout << "init\n";
        init(50,
             100,
             0.001,
             0.9,
             0.97,
             100,
             "startrek_data.txt",
             "lstm.dat");
    }
    else
    {
        train("startrek_data.txt", "lstm.dat");
    }
}

void init(int epochs,                         // -e
          int seq_length,                     // -s
          double rate,                        // -r
          double momentum,                    // -m
          double rate_decay,                  // -d
          int num_cells,                      // -c
          const std::string &data_file,       // -i
          const std::string &checkpoint_file) // -o
{
    std::vector<std::string> input_data = read_data(data_file);
    if (input_data.empty())
    {
        std::cerr << "no input\n";
        exit(EXIT_FAILURE);
    }

    uint32_t epoch = 0;
    uint32_t file = 0;
    std::vector<uint32_t> order(input_data.size());
    for (uint32_t i = 0; i < input_data.size(); ++i)
        order[i] = i;
    std::random_shuffle(order.begin(), order.end());

    TextMapper mapper(input_data);
    uint32_t num_input = mapper.num_classes();
    uint32_t num_output = mapper.num_classes();

    Layer L1 = lstm_layer(num_input, num_cells);
    Layer L2 = lstm_layer(num_input, num_cells);
    Layer L3 = lstm_layer(num_input, num_cells);
    lstm_init_layer(L1);
    lstm_init_layer(L2);
    lstm_init_layer(L3);

    Matrix Wyh(num_output, num_cells);
    rng::setnormal(Wyh, 0, std::sqrt(num_cells));
    Vector by(num_output);

    Layer D1 = lstm_layer(num_input, num_cells);
    Layer D2 = lstm_layer(num_input, num_cells);
    Layer D3 = lstm_layer(num_input, num_cells);
    Matrix dWyh = Matrix::Zero(num_output, num_cells);
    Vector dy = Vector::Zero(num_output);

    Layer M1 = lstm_layer(num_input, num_cells);
    Layer M2 = lstm_layer(num_input, num_cells);
    Layer M3 = lstm_layer(num_input, num_cells);
    Matrix mWyh = Matrix::Zero(num_output, num_cells);
    Vector my = Vector::Zero(num_output);

    checkpoint::save(checkpoint_file,
                     // static data
                     epochs,
                     seq_length,
                     rate,
                     momentum,
                     rate_decay,
                     num_cells,

                     // current iteration
                     epoch,
                     file,
                     order,

                     // params
                     L1.W, L1.b,
                     L2.W, L2.b,
                     L3.W, L3.b,
                     Wyh, by,

                     // gradients, momentum
                     D1.W, D1.b,
                     D2.W, D2.b,
                     D3.W, D3.b,
                     dWyh, dy,
                     M1.W, M1.b,
                     M2.W, M2.b,
                     M3.W, M3.b,
                     mWyh, my);
}

void train(const std::string &data_file, const std::string &checkpoint_file)
{
    std::vector<std::string> input_data = read_data(data_file);
    if (input_data.empty())
    {
        std::cerr << "no input\n";
        exit(EXIT_FAILURE);
    }
    TextMapper mapper(input_data);
    uint32_t num_input = mapper.num_classes();
    uint32_t num_output = mapper.num_classes();

    int epochs;
    int seq_length;
    int num_cells;

    double rate;
    double momentum;
    double rate_decay;

    int epoch;
    int file;
    std::vector<int> order;

    Layer L1;
    Layer L2;
    Layer L3;
    Matrix Wyh;
    Vector by;

    Layer D1;
    Layer D2;
    Layer D3;
    Matrix dWyh;
    Vector dy;

    Layer M1;
    Layer M2;
    Layer M3;
    Matrix mWyh;
    Vector my;

    checkpoint::load(checkpoint_file,
                     // static data
                     epochs,
                     seq_length,
                     rate,
                     momentum,
                     rate_decay,
                     num_cells,

                     // current iteration
                     epoch,
                     file,
                     order,

                     // params
                     L1.W, L1.b,
                     L2.W, L2.b,
                     L3.W, L3.b,
                     Wyh, by,

                     // gradients, momentum
                     D1.W, D1.b,
                     D2.W, D2.b,
                     D3.W, D3.b,
                     dWyh, dy,
                     M1.W, M1.b,
                     M2.W, M2.b,
                     M3.W, M3.b,
                     mWyh, my);

    // BPTT - track each state / output
    std::vector<Vector> states1(seq_length + 1);
    std::vector<Vector> states2(seq_length + 1);
    std::vector<Vector> states3(seq_length + 1);

    std::vector<Vector> outputs1(seq_length + 1);
    std::vector<Vector> outputs2(seq_length + 1);
    std::vector<Vector> outputs3(seq_length + 1);

    for (size_t i = 0; i < seq_length; ++i)
    {
        states1[i] = lstm_state(num_cells);
        states1[i] = lstm_state(num_cells);
        states1[i] = lstm_state(num_cells);
    }

    while (epoch < epochs)
    {
        while (file < order.size())
        {
            std::cout << epoch << " - " << file << "\n";
            size_t seq_idx = 0;
            /*
            Layer L = lstm_layer(2, 1);
            Vector state1 = lstm_state(1);
            Vector o = lstm_output(state1);
            return o[0];

            Vector x(2);
            x << 1, 0.5;
            Vector state2 = lstm_state(1);

            state1 << 0, 0, 0, 0, 0.5, 0.75;
            lstm_forwardpass(L, state1, x, state2);

            Vector h = lstm_output(state2);
            Vector dh(1);
            dh << 0.67;

            Vector d = lstm_state(1);
            d << 0.1, 0.2, 0.3, 0.4, -0.1, -0.2;

            Matrix dW = Matrix::Zero(4 * 1, 2 + 1);
            Vector dx = Vector::Zero(2);

            lstm_backwardpass(L, state2, x, state1, dh, dW, d, dx);
            */

            ++file;
            if (file < order.size())
            {
                checkpoint::save(checkpoint_file,
                                 // static data
                                 epochs,
                                 seq_length,
                                 rate,
                                 momentum,
                                 rate_decay,
                                 num_cells,

                                 // current iteration
                                 epoch,
                                 file,
                                 order,

                                 // params
                                 L1.W, L1.b,
                                 L2.W, L2.b,
                                 L3.W, L3.b,
                                 Wyh, by,

                                 // gradients, momentum
                                 D1.W, D1.b,
                                 D2.W, D2.b,
                                 D3.W, D3.b,
                                 dWyh, dy,
                                 M1.W, M1.b,
                                 M2.W, M2.b,
                                 M3.W, M3.b,
                                 mWyh, my);
            }
        }

        // finished epoch
        // save state
        ++epoch;
        file = 0;
        std::random_shuffle(order.begin(), order.end());
        checkpoint::save(checkpoint_file,
                         // static data
                         epochs,
                         seq_length,
                         rate,
                         momentum,
                         rate_decay,
                         num_cells,

                         // current iteration
                         epoch,
                         file,
                         order,

                         // params
                         L1.W, L1.b,
                         L2.W, L2.b,
                         L3.W, L3.b,
                         Wyh, by,

                         // gradients, momentum
                         D1.W, D1.b,
                         D2.W, D2.b,
                         D3.W, D3.b,
                         dWyh, dy,
                         M1.W, M1.b,
                         M2.W, M2.b,
                         M3.W, M3.b,
                         mWyh, my);
    }
}
