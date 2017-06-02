#include <iostream>
#include <fstream>
#include <unistd.h>

#include "lib/core.h"
#include "lib/lstm.h"
#include "lib/checkpoint.h"
#include "lib/read_data.h"
#include "lib/text_mapper.h"
#include "lib/rng.h"
#include "lib/optimize.h"
#include "lib/js.h"

void init(const int epochs,          // -E
          const int seq_length,      // -s
          const double rate,         // -r
          const double momentum,     // -m
          const double rate_decay,   // -d
          const double output_noise, // -g
          const double dropout,      // -D
          const int num_cells,       // -n
          const std::string &checkpoint_file,
          const std::string &data_file);

void train(const int from_epoch,               // -e
           const int from_file,                // -f
           const int epochs_override,          // -E
           const int seq_length_override,      // -s
           const double rate_override,         // -r
           const double momentum_override,     // -m
           const double rate_decay_override,   // -d
           const double output_noise_override, // -g
           const double dropout_override,      // -D
           const std::string &checkpoint_file,
           const std::string &data_file);

void sample(const int n,       // -n
            const double temp, // -t
            const std::string &checkpoint_file, const std::string &data_file);

void export_js(const std::string &checkpoint_file, const std::string &data_file, const std::string &output_file);

void print_usage(const char *execname);
void print_usage_init(const char *execname);
void print_usage_train(const char *execname);
void print_usage_sample(const char *execname);
void print_usage_export(const char *execname);

//
// main
// parse argv
// dispatch command function - init(), train(), sample()
//
int main(int argc, char **argv)
{

    if (argc < 2)
        print_usage(argv[0]);

    std::string cmd = argv[1];

    if (cmd == "init")
    {
        if (argc < 4)
            print_usage_init(argv[0]);

        std::string checkpoint_file = argv[argc - 2];
        std::string data_file = argv[argc - 1];

        int c;
        int epochs = 50;
        int seq_length = 100;
        double rate = 0.0005;
        double momentum = 0.9;
        double rate_decay = 0.975;
        double output_noise = 0.01;
        double dropout = 0.4;
        int num_cells = 100;
        while ((c = getopt(argc - 1, argv + 1, "E:s:r:m:d:g:D:n:")) != -1)
        {
            switch (c)
            {
            case 'E':
                epochs = std::stoi(optarg);
                if (epochs <= 0)
                {
                    std::cerr << "epochs must be > 0\n";
                    exit(1);
                }
                break;
            case 's':
                seq_length = std::stoi(optarg);
                if (seq_length <= 0)
                {
                    std::cerr << "sequence length must be > 0\n";
                    exit(1);
                }
                break;
            case 'r':
                rate = std::stod(optarg);
                break;
            case 'm':
                momentum = std::stod(optarg);
                break;
            case 'd':
                rate_decay = std::stod(optarg);
                break;
            case 'g':
                output_noise = std::stod(optarg);
                break;
            case 'D':
                dropout = std::stod(optarg);
                break;
            case 'n':
                num_cells = std::stoi(optarg);
                if (num_cells <= 0)
                {
                    std::cerr << "number of cells must be > 0\n";
                    exit(1);
                }
                break;
            case '?':
                print_usage_init(argv[0]);
            default:
                abort();
            }
        }
        std::cout << "Initializing with settings:\n"
                  << "epochs                " << epochs << "\n"
                  << "sequence length       " << seq_length << "\n"
                  << "learning rate         " << rate << "\n"
                  << "momentum              " << momentum << "\n"
                  << "learning rate decay   " << rate_decay << "\n"
                  << "output noise          " << output_noise << "\n"
                  << "dropout rate          " << dropout << "\n"
                  << "hidden layer size     " << num_cells << "\n";
        init(epochs,
             seq_length,
             rate,
             momentum,
             rate_decay,
             output_noise,
             dropout,
             num_cells,
             checkpoint_file,
             data_file);
    }
    else if (cmd == "train")
    {
        if (argc < 4)
            print_usage_init(argv[0]);

        std::string checkpoint_file = argv[argc - 2];
        std::string data_file = argv[argc - 1];

        int c;
        int from_epoch = -1;
        int from_file = -1;
        int epochs_override = -1;
        int seq_length_override = -1;
        double rate_override = -1;
        double momentum_override = -1;
        double rate_decay_override = -1;
        double output_noise_override = -1;
        double dropout_override = -1;

        while ((c = getopt(argc - 1, argv + 1, "e:f:E:s:r:m:d:g:D:n:")) != -1)
        {
            switch (c)
            {
            case 'e':
                from_epoch = std::stoi(optarg);
                if (from_epoch < 0)
                {
                    std::cerr << "starting epoch must be >= 0\n";
                    exit(1);
                }
                break;
            case 'f':
                from_file = std::stoi(optarg);
                if (from_file < 0)
                {
                    std::cerr << "starting file must be >= 0\n";
                    exit(1);
                }
                break;
            case 'E':
                epochs_override = std::stoi(optarg);
                if (epochs_override <= 0)
                {
                    std::cerr << "epochs must be > 0\n";
                    exit(1);
                }
                break;
            case 's':
                seq_length_override = std::stoi(optarg);
                if (seq_length_override <= 0)
                {
                    std::cerr << "sequence length must be > 0\n";
                    exit(1);
                }
                break;
            case 'r':
                rate_override = std::stod(optarg);
                break;
            case 'm':
                momentum_override = std::stod(optarg);
                break;
            case 'd':
                rate_decay_override = std::stod(optarg);
                break;
            case 'g':
                output_noise_override = std::stod(optarg);
                break;
            case 'D':
                dropout_override = std::stod(optarg);
                break;
            case '?':
                print_usage_init(argv[0]);
            default:
                abort();
            }
        }

        train(from_epoch,
              from_file,
              epochs_override,
              seq_length_override,
              rate_override,
              momentum_override,
              rate_decay_override,
              output_noise_override,
              dropout_override,
              checkpoint_file,
              data_file);
    }
    else if (cmd == "sample")
    {
        if (argc < 4)
            print_usage_sample(argv[0]);

        int n = 1024;
        double temp = 1;
        int c;
        while ((c = getopt(argc - 1, argv + 1, "t:n:")) != -1)
        {
            switch (c)
            {
            case 't':
                temp = std::stod(optarg);
                break;
            case 'n':
                n = std::stoi(optarg);
                if (n <= 0)
                {
                    std::cerr << "sample size must be > 0\n";
                    exit(1);
                }
                break;
            case '?':
                print_usage_sample(argv[0]);
            default:
                abort();
            }
        }
        std::string checkpoint_file = argv[argc - 2];
        std::string data_file = argv[argc - 1];
        sample(n, temp, checkpoint_file, data_file);
    }
    else if (cmd == "export")
    {
        if (argc < 5)
            print_usage_export(argv[0]);
        std::string checkpoint_file = argv[argc - 3];
        std::string data_file = argv[argc - 2];
        std::string output_file = argv[argc - 1];
        export_js(checkpoint_file, data_file, output_file);
    }
    else
    {
        print_usage(argv[0]);
    }
}

//
//
// init
//
//

void init(const int epochs,          // -e
          const int seq_length,      // -s
          const double rate,         // -r
          const double momentum,     // -m
          const double rate_decay,   // -d
          const double output_noise, // -g
          const double dropout,      // -D
          const int num_cells,       // -n
          const std::string &checkpoint_file,
          const std::string &data_file)
{
    std::cout << "Reading input from " << data_file << "\n";
    std::vector<std::string *> input_data = read_data(data_file);

    if (input_data.empty())
    {
        std::cerr << "no input\n";
        exit(1);
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
    Layer L2 = lstm_layer(num_cells, num_cells);
    Layer L3 = lstm_layer(num_cells, num_cells);
    lstm_init_layer(L1);
    lstm_init_layer(L2);
    lstm_init_layer(L3);

    Matrix Wyh(num_output, num_cells);
    rng::setnormal(Wyh, 0, 1.0 / std::sqrt(num_cells));
    Vector by = Vector::Random(num_output) * 0.1;

    Layer M1 = lstm_layer(num_input, num_cells);
    Layer M2 = lstm_layer(num_cells, num_cells);
    Layer M3 = lstm_layer(num_cells, num_cells);
    Matrix mWyh = Matrix::Zero(num_output, num_cells);
    Vector my = Vector::Zero(num_output);

    std::cout << "Saving to " << checkpoint_file << "\n";
    checkpoint::save(checkpoint_file,
                     // static data
                     epochs,
                     seq_length,
                     rate,
                     momentum,
                     rate_decay,
                     output_noise,
                     dropout,
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
                     M1.W, M1.b,
                     M2.W, M2.b,
                     M3.W, M3.b,
                     mWyh, my);

    std::for_each(input_data.begin(), input_data.end(), [](std::string *chunk) { delete chunk; });
    std::cout << "Done\n";
}

//
//
// train
//
//

void train(const int from_epoch,               // -e
           const int from_file,                // -f
           const int epochs_override,          // -E
           const int seq_length_override,      // -s
           const double rate_override,         // -r
           const double momentum_override,     // -m
           const double rate_decay_override,   // -d
           const double output_noise_override, // -g
           const double dropout_override,      // -D
           const std::string &checkpoint_file,
           const std::string &data_file)
{
    std::cout << "Reading input from " << data_file << "\n";
    std::vector<std::string *> input_data = read_data(data_file);
    if (input_data.empty())
    {
        std::cerr << "no input\n";
        exit(1);
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
    double output_noise;
    double dropout;

    int epoch;
    int file;
    std::vector<int> order;

    Layer L1;
    Layer L2;
    Layer L3;
    Matrix Wyh;
    Vector by;

    Layer M1;
    Layer M2;
    Layer M3;
    Matrix mWyh;
    Vector my;

    std::cout << "Loading from " << checkpoint_file << "\n";
    checkpoint::load(checkpoint_file,
                     // static data, hyperparams
                     epochs,
                     seq_length,
                     rate,
                     momentum,
                     rate_decay,
                     output_noise,
                     dropout,
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

                     // momentum
                     M1.W, M1.b,
                     M2.W, M2.b,
                     M3.W, M3.b,
                     mWyh, my);

    L1.num_input = num_input;
    L1.num_cells = num_cells;
    L2.num_input = L2.num_cells = num_cells;
    L3.num_input = L3.num_cells = num_cells;

    // Track accumulated gradients
    Layer G1 = lstm_layer(num_input, num_cells);
    Layer G2 = lstm_layer(num_cells, num_cells);
    Layer G3 = lstm_layer(num_cells, num_cells);
    Matrix gWyh = Matrix::Zero(num_output, num_cells);
    Vector gy = Vector::Zero(num_output);

    // BPTT - track each state / output
    std::vector<Vector> states1(seq_length + 1);
    std::vector<Vector> states2(seq_length + 1);
    std::vector<Vector> states3(seq_length + 1);

    std::vector<Vector> outputs(seq_length + 1);

    Vector x(num_input);
    Vector y_(num_output);
    Vector p(num_output);

    // Track intermediate gradients
    Gradients D1 = lstm_gradients(num_input, num_cells);
    Gradients D2 = lstm_gradients(num_cells, num_cells);
    Gradients D3 = lstm_gradients(num_cells, num_cells);
    // dWyh not needed
    Vector dy(num_output);

    Vector dh(num_cells);

    // apply command line overrides
    if (from_epoch >= 0)
        epoch = from_epoch;
    if (from_file >= 0)
        file = from_file;
    if (epochs_override >= 0)
        epochs = epochs_override;
    if (seq_length_override >= 0)
        seq_length = seq_length_override;
    if (rate_override >= 0)
        rate = rate_override;
    if (momentum_override >= 0)
        momentum = momentum_override;
    if (rate_decay_override >= 0)
        rate_decay = rate_decay_override;
    if (output_noise_override >= 0)
        output_noise = output_noise_override;
    if (dropout_override >= 0)
        dropout = dropout_override;

    std::cout << "Training with settings:\n"
              << "epochs                " << epochs << "\n"
              << "sequence length       " << seq_length << "\n"
              << "learning rate         " << rate << "\n"
              << "momentum              " << momentum << "\n"
              << "learning rate decay   " << rate_decay << "\n"
              << "output noise          " << output_noise << "\n"
              << "dropout rate          " << dropout << "\n"
              << "hidden layer size     " << num_cells << "\n"
              << "input/output size     " << mapper.num_classes() << "\n";

    int total_bytes = 10000;

    // loop through epochs
    while (epoch < epochs)
    {
        double decayed_rate = rate * std::pow(rate_decay, epoch);

        // epoch - loop through files
        while (file < order.size())
        {
            std::cout << epoch << " - " << file << "\n";
            double error = 0;
            int idx = 0;
            int seq_offset = 0;
            std::string *chunk = input_data[order[file]];

            // file - BPTT over each subsequence
            while (seq_offset < chunk->size())
            {
                // Reset accumulated gradients
                G1.W.setZero();
                G1.b.setZero();
                G2.W.setZero();
                G2.b.setZero();
                G3.W.setZero();
                G3.b.setZero();
                gWyh.setZero();
                gy.setZero();

                // Occasionally reset state
                if (total_bytes >= 10000)
                {
                    std::cout << (100 * seq_offset / chunk->size()) << "%\n";
                    for (int i = 0; i < seq_length + 1; ++i)
                    {
                        states1[i] = lstm_state(num_cells);
                        states2[i] = lstm_state(num_cells);
                        states3[i] = lstm_state(num_cells);
                    }
                    total_bytes = 0;
                    checkpoint::save(checkpoint_file,
                                     // static data, hyperparams
                                     epochs,
                                     seq_length,
                                     rate,
                                     momentum,
                                     rate_decay,
                                     output_noise,
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

                                     // momentum
                                     M1.W, M1.b,
                                     M2.W, M2.b,
                                     M3.W, M3.b,
                                     mWyh, my);
                }

                int seq_end = seq_offset + seq_length;
                if (seq_end >= chunk->size())
                    seq_end = chunk->size() - 1;

                total_bytes += seq_end - seq_offset;

                // BPTT - loop through sequence
                for (int i = seq_offset; i < seq_end; ++i)
                {
                    mapper.to_onehot((*chunk)[i], x);

                    int idx_next = (idx + 1) % (seq_length + 1);
                    lstm_forwardpass(L1, states1[idx], x, states1[idx_next]);
                    lstm_forwardpass(L2, states2[idx], lstm_output(states1[idx_next]), states2[idx_next]);
                    lstm_forwardpass(L3, states3[idx], lstm_output(states2[idx_next]), states3[idx_next]);

                    outputs[idx] = Wyh * lstm_output(states3[idx_next]) + by;

                    // regularization - apply gaussian noise to output
                    for (uint32_t j = 0; j < num_output; ++j)
                        outputs[idx][j] += rng::normal(0, output_noise);

                    idx = idx_next;
                }

                // Reset intermediate gradients
                D1.W.setZero();
                D1.S.setZero();
                D2.W.setZero();
                D2.S.setZero();
                D3.W.setZero();
                D3.S.setZero();
                dy.setZero();

                for (int i = seq_end - 1; i >= seq_offset; --i)
                {
                    size_t idx_next = idx;
                    idx = (idx_next - 1) % (seq_length + 1);
                    bool has_next = i < seq_end - 1;

                    mapper.to_onehot((*chunk)[i], x);
                    mapper.to_onehot((*chunk)[i + 1], y_);
                    error += softmax_cross_entropy_onehot(y_, outputs[idx], p, dy);

                    dh = Wyh.transpose() * dy;

                    lstm_backwardpass(has_next, L3, states3[idx_next], lstm_output(states2[idx_next]), states3[idx], dh, D3, dh);
                    lstm_backwardpass(has_next, L2, states2[idx_next], lstm_output(states1[idx_next]), states2[idx], dh, D2, dh);
                    lstm_backwardpass(has_next, L1, states1[idx_next], x, states1[idx], dh, D1);

                    // Accumulate gradients
                    G1.W += D1.W;
                    G1.b += D1.S.segment(0, 4 * num_cells);
                    G2.W += D2.W;
                    G2.b += D2.S.segment(0, 4 * num_cells);
                    G3.W += D3.W;
                    G3.b += D3.S.segment(0, 4 * num_cells);
                    gWyh += dy * lstm_output(states3[idx_next]).transpose();
                    gy += dy;
                }
                seq_offset += seq_length;

                optimize::nesterov(decayed_rate, momentum,
                                   // layer 1
                                   L1.W, G1.W, M1.W,
                                   L1.b, G1.b, M1.b,

                                   // layer 2
                                   L2.W, G2.W, M2.W,
                                   L2.b, G2.b, M2.b,

                                   // layer 3
                                   L3.W, G3.W, M3.W,
                                   L3.b, G3.b, M3.b,

                                   // output
                                   Wyh, gWyh, mWyh,
                                   by, gy, my);
            }

            error /= chunk->size();

            ++file;
            if (file < order.size())
            {
                checkpoint::save(checkpoint_file,
                                 // static data, hyperparams
                                 epochs,
                                 seq_length,
                                 rate,
                                 momentum,
                                 rate_decay,
                                 output_noise,
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

                                 // momentum
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
                         // static data, hyperparams
                         epochs,
                         seq_length,
                         rate,
                         momentum,
                         rate_decay,
                         output_noise,
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

                         // momentum
                         M1.W, M1.b,
                         M2.W, M2.b,
                         M3.W, M3.b,
                         mWyh, my);
    }

    std::cout << "Finished\n";
    std::for_each(input_data.begin(), input_data.end(), [](std::string *chunk) { delete chunk; });
}

//
//
// sample
//
//
void sample(const int n, const double temp, const std::string &checkpoint_file, const std::string &data_file)
{
    std::vector<std::string *> input_data = read_data(data_file);
    if (input_data.empty())
    {
        std::cerr << "no input\n";
        exit(1);
    }
    TextMapper mapper(input_data);
    uint32_t num_input = mapper.num_classes();
    uint32_t num_output = mapper.num_classes();
    int _i;
    double _f;
    int num_cells;

    int epoch;
    int file;
    std::vector<int> order;

    Layer L1;
    Layer L2;
    Layer L3;
    Matrix Wyh;
    Vector by;

    checkpoint::load(checkpoint_file,
                     // static data - we just need num_cells
                     _i,
                     _i,
                     _f,
                     _f,
                     _f,
                     _f,
                     num_cells,

                     // current iteration - don't need this
                     _i,
                     _i,
                     order,

                     // params - still need these
                     L1.W, L1.b,
                     L2.W, L2.b,
                     L3.W, L3.b,
                     Wyh, by);

    L1.num_input = num_input;
    L1.num_cells = num_cells;
    L2.num_input = L2.num_cells = num_cells;
    L3.num_input = L3.num_cells = num_cells;

    Vector state1 = lstm_state(num_cells);
    Vector state2 = lstm_state(num_cells);
    Vector state3 = lstm_state(num_cells);

    Vector x(num_input);
    Vector y(num_output);
    Vector p(num_output);

    char c = '^';
    for (int i = 0; i < n; ++i)
    {
        mapper.to_onehot(c, x);
        lstm_forwardpass(L1, state1, x, state1);
        lstm_forwardpass(L2, state2, lstm_output(state1), state2);
        lstm_forwardpass(L3, state3, lstm_output(state2), state3);
        y = (Wyh * lstm_output(state3) + by) / temp;
        softmax_activation(y, p);
        c = mapper.from_dist(p);
        std::cout << c;
    }
}

void export_js(const std::string &checkpoint_file, const std::string &data_file, const std::string &output_file)
{
    std::vector<std::string *> input_data = read_data(data_file);
    if (input_data.empty())
    {
        std::cerr << "no input\n";
        exit(1);
    }
    TextMapper mapper(input_data);
    uint32_t num_input = mapper.num_classes();
    uint32_t num_output = mapper.num_classes();
    int _i;
    double _f;
    int num_cells;

    int epoch;
    int file;
    std::vector<int> order;

    Layer L1;
    Layer L2;
    Layer L3;
    Matrix Wyh;
    Vector by;

    checkpoint::load(checkpoint_file,
                     // static data - we just need num_cells
                     _i,
                     _i,
                     _f,
                     _f,
                     _f,
                     _f,
                     num_cells,

                     // current iteration - don't need this
                     _i,
                     _i,
                     order,

                     // params - still need these
                     L1.W, L1.b,
                     L2.W, L2.b,
                     L3.W, L3.b,
                     Wyh, by);
    js::outfile out(output_file);

    out << "var la = linearAlgebra();\n"
        << "var Matrix = la.Matrix;\n"
        << "var Vector = la.Vector;\n"
        << "var num_cells = " << num_cells << ";\n"
        << "var num_input = " << num_input << ";\n"
        << "var num_output = " << num_output << ";\n"
        << "\n"
        << "var encoder = ";
    char sep = '[';
    for (int i = 0; i < num_input; ++i)
    {
        out << sep << (int)mapper.encode(i);
        sep = ',';
    }
    out << "];\n"
        << "var decoder = ";
    sep = '[';
    for (int i = 0; i < num_input; ++i)
    {
        out << sep << (int)((uint8_t)mapper.decode(i));
        sep = ',';
    }
    out << "];\n"
        << "var encode = function(c) { return encoder[c.charCodeAt(0)]; }\n"
        << "var decode = function(i) { return String.fromCharCode(decoder[i]); }\n"
        << "\n"
        << "var L1 = {\n"
        << "  W: " << L1.W << ",\n"
        << "  b: " << L1.b << "\n"
        << "};\n"
        << "var L2 = {\n"
        << "  W: " << L2.W << ",\n"
        << "  b: " << L2.b << "\n"
        << "};\n"
        << "var L3 = {\n"
        << "  W: " << L3.W << ",\n"
        << "  b: " << L3.b << "\n"
        << "};\n"
        << "var Wyh = " << Wyh << ";\n"
        << "var by = " << by << ";\n"
        << "\n"
        << "";
}

//
//
// PRINT USAGE
//
void print_usage(const char *execname)
{
    std::cerr << "usage:\n"
              << execname << " init -esrmdn checkpoint_file data_file\n"
              << execname << " train checkpoint_file data_file\n"
              << execname << " sample checkpoint_file data_file\n"
              << execname << " export checkpoint_file data_file output_file\n";
    exit(1);
}

void print_usage_init(const char *execname)
{
    std::cerr << "usage: " << execname << " init [-Esrmdn] checkpoint_file data_file\n"
              << "flag  option              default\n"
              << "-E:   epochs              50\n"
              << "-s:   sequence length     100\n"
              << "-r:   learning rate       0.0005\n"
              << "-m:   momentum            0.9\n"
              << "-d:   learning rate decay 0.975\n"
              << "-g:   output noise        0.01\n"
              << "-D:   dropout rate        0.4\n"
              << "-n:   cells per layer     100\n";
    exit(1);
}

void print_usage_train(const char *execname)
{
    std::cerr << "usage: " << execname << " train [-efEsrmd] checkpoint_file data_file\n"
              << "flag  option              default\n"
              << "-e:   starting epoch      [current epoch]\n"
              << "-f:   starting file index [current file index]\n"
              << "-E:   epochs              [saved value]\n"
              << "-s:   sequence length     [saved value]\n"
              << "-r:   learning rate       [saved value]\n"
              << "-m:   momentum            [saved value]\n"
              << "-d:   learning rate decay [saved value]\n"
              << "-g:   output noise        [saved value]\n"
              << "-D:   dropout rate        [saved value]\n";
    exit(1);
}

void print_usage_sample(const char *execname)
{
    std::cerr << "usage: " << execname << " sample [-nt] checkpoint_file data_file\n"
              << "flag  option              default\n"
              << "-n:   number of bytes     1024\n"
              << "-t:   softmax temperature 1.0\n";
    exit(1);
}

void print_usage_export(const char *execname)
{
    std::cerr << "usage: " << execname << " export checkpoint_file data_file output_file\n";
    exit(1);
}
