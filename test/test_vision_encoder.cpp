// test_vision_encoder.cpp
// Verifies vision_encoder output shape and numerical validity,
// and saves the result as a binary file for external comparison.
//
// Usage:
//   ./test_vision_encoder <mmproj.gguf> <output.bin>

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "coalsack/llm/vision_encoder.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <mmproj.gguf> <output.bin>\n";
    return 1;
  }

  spdlog::set_level(spdlog::level::trace);

  std::string mmproj_path = argv[1];
  std::string out_path = argv[2];

  coalsack::vision_encoder encoder;
  if (!encoder.load(mmproj_path)) {
    std::cerr << "ERROR: Failed to load mmproj from: " << mmproj_path << "\n";
    return 1;
  }

  const int n_tokens = encoder.n_output_tokens();
  const int proj_dim = encoder.get_config().proj_dim;
  const int H = encoder.get_config().image_size;
  const int W = encoder.get_config().image_size;

  std::cout << "=== Vision Encoder Test ===\n\n";
  std::cout << "Model: n_output_tokens=" << n_tokens << "  proj_dim=" << proj_dim
            << "  image_size=" << W << "x" << H << "\n\n";

  // Synthetic image: all pixels = 127
  std::vector<uint8_t> image(H * W * 3, 127);

  std::cout << "Encoding synthetic image (" << W << "x" << H << ", all=127) ...\n";
  coalsack::dynamic_tensor result = encoder.encode(image.data(), W, H);

  // --- Shape check ---
  if (result.ndim() != 2 || result.dim(0) != n_tokens || result.dim(1) != proj_dim) {
    std::cerr << "ERROR: Unexpected output shape: [" << result.dim(0) << ", " << result.dim(1)
              << "]  expected [" << n_tokens << ", " << proj_dim << "]\n";
    return 1;
  }
  std::cout << "Shape check PASSED: [" << n_tokens << ", " << proj_dim << "]\n";

  // --- NaN / Inf check ---
  const float* out = result.data_ptr<float>();
  const int total = n_tokens * proj_dim;
  int nan_count = 0, inf_count = 0;
  for (int i = 0; i < total; ++i) {
    if (std::isnan(out[i]))
      ++nan_count;
    else if (std::isinf(out[i]))
      ++inf_count;
  }
  if (nan_count > 0 || inf_count > 0) {
    std::cerr << "ERROR: Output contains NaN=" << nan_count << " Inf=" << inf_count << "\n";
    return 1;
  }
  std::cout << "NaN/Inf check PASSED\n";

  // --- Save binary ---
  {
    std::ofstream f(out_path, std::ios::binary);
    if (!f) {
      std::cerr << "ERROR: Cannot write to: " << out_path << "\n";
      return 1;
    }
    // 8-byte header: n_tokens, proj_dim
    int32_t hdr[2] = {n_tokens, proj_dim};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(out), total * sizeof(float));
    std::cout << "\nSaved " << total << " floats to: " << out_path << "\n";
  }

  std::cout << "\n=== Test Complete ===\n";
  return 0;
}
