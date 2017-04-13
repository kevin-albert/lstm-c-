#ifndef rng_h
#define rng_h

#include "core.h"

namespace rng
{
float uniform(float min, float max);
float normal(float mean, float stddev);
void setnormal(Matrix &m, float mean, float stddev);
}

#endif