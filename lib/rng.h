#ifndef rng_h
#define rng_h

#include "core.h"

namespace rng
{
float uniform(float min, float max);
float normal(float mean, float stddev);
int randint(int max);
void setnormal(Matrix &m, float mean, float stddev);
}

#endif