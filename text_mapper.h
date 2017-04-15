#ifndef text_mapper_h
#define text_mapper_h

#include <vector>
#include <string>
#include "core.h"

class TextMapper
{
public:
  TextMapper(std::vector<std::string *> &data);
  void to_onehot(const unsigned char, Vector &) const;
  unsigned char from_onehot(Vector &) const;
  unsigned char from_dist(Vector &) const;
  uint32_t num_classes() const { return decoder.size(); }

private:
  std::vector<int> encoder;
  std::vector<char> decoder;
};

#endif