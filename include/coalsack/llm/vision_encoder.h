#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "coalsack/tensor/dynamic_tensor.h"

namespace coalsack {

// Vision encoder: patch embedding ViT + pixel shuffle + MLP projector.
// Reads weights from a mmproj GGUF file.
class vision_encoder {
 public:
  struct config {
    int image_size = 336;
    int patch_size = 14;
    int n_embd = 1408;
    int n_head = 16;
    int n_layer = 34;
    int ffn_dim = 5632;
    int proj_dim = 5120;           // output/LLM hidden dim
    int scale_factor = 2;          // pixel shuffle scale
    int proj_intermediate = 4096;  // MLP intermediate dim in projector
    float rope_theta = 10000.0f;
    float ln_eps = 1e-5f;
    float image_mean = 0.5f;
    float image_std = 0.5f;
  };

  vision_encoder();
  ~vision_encoder();

  // Load mmproj GGUF file.
  bool load(const std::string& mmproj_path);
  bool is_loaded() const;

  // Encode RGB image (HWC uint8 layout: data[y * width*3 + x*3 + c])
  // width and height must equal config.image_size (336)
  // Returns float32 tensor of shape [n_output_tokens, proj_dim] = [144, 5120]
  dynamic_tensor encode(const uint8_t* data, int width, int height);

  // Encode pre-normalized float32 image (HWC layout, values normalised already)
  dynamic_tensor encode_f32(const float* data, int width, int height);

  // Number of output vision tokens (144 for 336x336 with scale=2)
  int n_output_tokens() const;

  // Loaded config (available after load())
  const config& get_config() const;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl_;
};

}  // namespace coalsack
