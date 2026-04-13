/// @file types.h
/// @brief Minimal math types used across the framework.
/// @ingroup core_graph
#pragma once

namespace coalsack {

/// @brief 3-component floating-point vector.
struct vec3 {
  float x, y, z;
};

/// @brief 4x4 column-major floating-point matrix.
struct mat4 {
  float data[16];
};

}  // namespace coalsack
