#include <algorithm>
#include "text_mapper.h"
#include "rng.h"

TextMapper::TextMapper(std::vector<std::string> &data) : encoder(255)
{
    std::fill(encoder.begin(), encoder.end(), -1);

    std::for_each(data.begin(), data.end(), [this](const std::string &file) {
        std::for_each(file.begin(), file.end(), [this](char c) {
            if (encoder[c] == -1)
            {
                encoder[c] = decoder.size();
                decoder.push_back(c);
            }
        });
    });
}

void TextMapper::to_onehot(const char c, Vector &A) const
{
    A.resize(num_classes());
    A.setZero();
    int idx = encoder[c];
    if (idx >= 0)
    {
        A[idx] = 1;
    }
    else
    {
        abort();
    }
}

char TextMapper::from_onehot(Vector &A) const
{
    float maxv = A[0];
    int maxi = 0;
    for (int i = 1; i < A.size(); ++i)
    {
        if (A[i] > maxv)
        {
            maxv = A[i];
            maxi = i;
        }
    }

    return decoder[maxi];
}

char TextMapper::from_dist(Vector &p) const
{
    float sum = 0;
    float r = rng::normal(0, 1);
    for (int i = 0; i < p.size(); ++i)
    {
        sum += p[i];
        if (sum >= r)
        {
            return decoder[i];
        }
    }
    return decoder[p.size() - 1];
}
