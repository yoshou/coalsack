/// @file image_message.h
/// @brief graph_message wrapper for an image.
/// @ingroup image
#pragma once

#include <string>

#include "coalsack/core/graph_message.h"
#include "coalsack/image/image.h"

namespace coalsack {
/// @brief Message carrying a single @c image payload.
class image_message : public graph_message {
  image img;

 public:
  image_message() : img() {}

  void set_image(const image &img) { this->img = img; }
  void set_image(image &&img) { this->img = std::move(img); }
  const image &get_image() const { return img; }
  static std::string get_type() { return "image"; }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(img);
  }
};
}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_MESSAGE(coalsack::image_message, coalsack::graph_message)
