#pragma once

#include <array>

#include "coalsack/math/types.h"

namespace coalsack {

struct camera_t {
  int width;
  int height;
  float ppx;
  float ppy;
  float fx;
  float fy;
  std::array<float, 5> coeffs;
  mat4 pose;
};

}  // namespace coalsack
