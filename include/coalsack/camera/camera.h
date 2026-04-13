/// @file camera.h
/// @brief Camera intrinsic and extrinsic parameter structure.
/// @ingroup core_graph
#pragma once

#include <array>

#include "coalsack/math/types.h"

namespace coalsack {

/// @brief Camera intrinsic parameters plus extrinsic pose.
/// @details Stores focal length (fx, fy), principal point (ppx, ppy), distortion
///          coefficients, and a 4x4 pose matrix.
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
