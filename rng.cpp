#include <random>
#include <limits>
#include "rng.h"

namespace rng
{

std::default_random_engine gen;
std::uniform_real_distribution<float> distu(0, 1);
std::normal_distribution<float> distn(0, 1);
std::uniform_int_distribution<> disti(0, std::numeric_limits<int>::max());

float uniform(float min, float max) { return distu(gen) * (max - min) + min; }

float normal(float mean, float stddev) { return distn(gen) * stddev + mean; }

int randint(int max) { return disti(gen) % max; }

void setnormal(Matrix &m, float mean, float stddev)
{
    for (uint32_t i = 0; i < m.size(); ++i)
        m.data()[i] = normal(mean, stddev);
}
}
