#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gguf_types.h"

namespace coalsack {

// Block sizes for various quantization formats
constexpr int QK4_0 = 32;
constexpr int QK4_1 = 32;
constexpr int QK5_0 = 32;
constexpr int QK5_1 = 32;
constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int K_SCALE_SIZE = 12;
constexpr int QK_MXFP4 = 32;

// Convert fp16 (stored as uint16_t) to float32
inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exponent = (h >> 10) & 0x1F;
  uint32_t mantissa = h & 0x3FF;

  if (exponent == 0) {
    if (mantissa == 0) {
      // Zero
      uint32_t result = sign;
      float f;
      std::memcpy(&f, &result, sizeof(f));
      return f;
    } else {
      // Subnormal
      while (!(mantissa & 0x400)) {
        mantissa <<= 1;
        exponent--;
      }
      exponent++;
      mantissa &= ~0x400;
      exponent = exponent + (127 - 15);
      uint32_t result = sign | (exponent << 23) | (mantissa << 13);
      float f;
      std::memcpy(&f, &result, sizeof(f));
      return f;
    }
  } else if (exponent == 31) {
    // Inf or NaN
    uint32_t result = sign | 0x7F800000 | (mantissa << 13);
    float f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
  } else {
    // Normal number
    exponent = exponent + (127 - 15);
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
  }
}

// Dequantize Q4_0 block
// Block structure: fp16 d (scale), QK4_0/2 bytes of 4-bit quants
inline void dequantize_block_q4_0(const uint8_t* src, float* dst, int64_t k) {
  const int nb = k / QK4_0;

  for (int i = 0; i < nb; ++i) {
    const uint16_t* d_ptr = reinterpret_cast<const uint16_t*>(src);
    float d = fp16_to_fp32(*d_ptr);
    src += 2;

    for (int j = 0; j < QK4_0 / 2; ++j) {
      uint8_t byte = src[j];
      int q0 = (byte & 0x0F) - 8;
      int q1 = ((byte >> 4) & 0x0F) - 8;
      dst[i * QK4_0 + j * 2 + 0] = d * q0;
      dst[i * QK4_0 + j * 2 + 1] = d * q1;
    }
    src += QK4_0 / 2;
  }
}

// Dequantize Q5_0 block
// Block structure: fp16 d, 4 bytes qh (high bits), QK5_0/2 bytes qs (low 4 bits)
inline void dequantize_block_q5_0(const uint8_t* src, float* dst, int64_t k) {
  const int nb = k / QK5_0;

  for (int i = 0; i < nb; ++i) {
    const uint16_t* d_ptr = reinterpret_cast<const uint16_t*>(src);
    float d = fp16_to_fp32(*d_ptr);
    src += 2;

    const uint8_t* qh = src;
    src += 4;
    const uint8_t* qs = src;
    src += QK5_0 / 2;

    uint32_t qh_bits;
    std::memcpy(&qh_bits, qh, 4);

    for (int j = 0; j < QK5_0 / 2; ++j) {
      uint8_t byte = qs[j];
      int q0 = (byte & 0x0F);
      int q1 = ((byte >> 4) & 0x0F);

      // Add 5th bit from qh
      q0 |= ((qh_bits >> (j * 2)) & 1) << 4;
      q1 |= ((qh_bits >> (j * 2 + 1)) & 1) << 4;

      // Subtract 16 to center around 0
      q0 -= 16;
      q1 -= 16;

      dst[i * QK5_0 + j * 2 + 0] = d * q0;
      dst[i * QK5_0 + j * 2 + 1] = d * q1;
    }
  }
}

// Dequantize Q8_0 block
// Block structure: fp16 d, QK8_0 int8 quants
inline void dequantize_block_q8_0(const uint8_t* src, float* dst, int64_t k) {
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; ++i) {
    const uint16_t* d_ptr = reinterpret_cast<const uint16_t*>(src);
    float d = fp16_to_fp32(*d_ptr);
    src += 2;

    const int8_t* qs = reinterpret_cast<const int8_t*>(src);
    for (int j = 0; j < QK8_0; ++j) {
      dst[i * QK8_0 + j] = d * qs[j];
    }
    src += QK8_0;
  }
}

// Dequantize Q4_K block (super-block)
// Block structure: fp16 d, fp16 dmin, 12 bytes scales, QK_K/2 bytes quants
inline void dequantize_block_q4_K(const uint8_t* src, float* dst, int64_t k) {
  const int nb = k / QK_K;

  for (int i = 0; i < nb; ++i) {
    const uint16_t* dm = reinterpret_cast<const uint16_t*>(src);
    float d = fp16_to_fp32(dm[0]);
    float dmin = fp16_to_fp32(dm[1]);
    src += 4;

    const uint8_t* scales = src;
    src += K_SCALE_SIZE;

    const uint8_t* qs = src;
    src += QK_K / 2;

    // Decode scales (6-bit quantized)
    uint8_t sc[8], m[8];
    for (int j = 0; j < 4; ++j) {
      sc[j] = scales[j] & 0x3F;
      m[j] = scales[j + 4] & 0x3F;
    }
    for (int j = 0; j < 4; ++j) {
      sc[j + 4] = ((scales[j] >> 6) & 0x03) | (((scales[j + 4] >> 6) & 0x03) << 2) |
                  (((scales[j + 8] >> (j * 2)) & 0x03) << 4);
      m[j + 4] = ((scales[j] >> 4) & 0x0C) | (((scales[j + 4] >> 4) & 0x0C) << 2) |
                 (((scales[j + 8] >> (j * 2 + 4)) & 0x03) << 4);
    }

    // Dequantize
    for (int j = 0; j < QK_K / 2; ++j) {
      int block_idx = j / 16;
      float scale = d * sc[block_idx];
      float min = dmin * m[block_idx];

      uint8_t byte = qs[j];
      int q0 = byte & 0x0F;
      int q1 = (byte >> 4) & 0x0F;

      dst[i * QK_K + j * 2 + 0] = scale * q0 - min;
      dst[i * QK_K + j * 2 + 1] = scale * q1 - min;
    }
  }
}

// Dequantize Q6_K block (super-block)
// Block structure: QK_K/2 bytes ql, QK_K/4 bytes qh, QK_K/16 bytes scales, fp16 d
inline void dequantize_block_q6_K(const uint8_t* src, float* dst, int64_t k) {
  const int nb = k / QK_K;

  for (int i = 0; i < nb; ++i) {
    const uint8_t* ql = src;
    src += QK_K / 2;

    const uint8_t* qh = src;
    src += QK_K / 4;

    const int8_t* scales = reinterpret_cast<const int8_t*>(src);
    src += QK_K / 16;

    const uint16_t* d_ptr = reinterpret_cast<const uint16_t*>(src);
    float d = fp16_to_fp32(*d_ptr);
    src += 2;

    for (int n = 0; n < QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        int is = n / 16 + l / 16;

        uint8_t q_low = ql[n / 2 + l];
        uint8_t q_high = qh[n / 4 + l % 32];

        int q0 = (q_low & 0x0F) | (((q_high >> (2 * (l / 16))) & 0x03) << 4);
        int q1 = ((q_low >> 4) & 0x0F) | (((q_high >> (2 * (l / 16) + 4)) & 0x03) << 4);

        // Center around 0 (subtract 32)
        q0 -= 32;
        q1 -= 32;

        float scale = d * scales[is];
        dst[i * QK_K + n + l] = scale * q0;
        dst[i * QK_K + n + l + 32] = scale * q1;
      }
    }
  }
}

// Dequantize Q5_K block (super-block)
// Block structure: fp16 d, fp16 dmin, 12 bytes scales, QK_K/8 bytes qh, QK_K/2 bytes qs
inline void dequantize_block_q5_K(const uint8_t* src, float* dst, int64_t k) {
  const int nb = k / QK_K;

  for (int i = 0; i < nb; ++i) {
    const uint16_t* dm = reinterpret_cast<const uint16_t*>(src);
    float d = fp16_to_fp32(dm[0]);
    float dmin = fp16_to_fp32(dm[1]);
    src += 4;

    const uint8_t* scales = src;
    src += K_SCALE_SIZE;

    const uint8_t* qh = src;
    src += QK_K / 8;

    const uint8_t* qs = src;
    src += QK_K / 2;

    // Decode scales (same as Q4_K)
    uint8_t sc[8], m[8];
    for (int j = 0; j < 4; ++j) {
      sc[j] = scales[j] & 0x3F;
      m[j] = scales[j + 4] & 0x3F;
    }
    for (int j = 0; j < 4; ++j) {
      sc[j + 4] = ((scales[j] >> 6) & 0x03) | (((scales[j + 4] >> 6) & 0x03) << 2) |
                  (((scales[j + 8] >> (j * 2)) & 0x03) << 4);
      m[j + 4] = ((scales[j] >> 4) & 0x0C) | (((scales[j + 4] >> 4) & 0x0C) << 2) |
                 (((scales[j + 8] >> (j * 2 + 4)) & 0x03) << 4);
    }

    // Dequantize
    for (int j = 0; j < QK_K / 2; ++j) {
      int block_idx = j / 16;
      float scale = d * sc[block_idx];
      float min = dmin * m[block_idx];

      uint8_t byte = qs[j];
      int hbit0 = (qh[j / 4] >> ((j % 4) * 2)) & 1;
      int hbit1 = (qh[j / 4] >> ((j % 4) * 2 + 1)) & 1;

      int q0 = (byte & 0x0F) | (hbit0 << 4);
      int q1 = ((byte >> 4) & 0x0F) | (hbit1 << 4);

      dst[i * QK_K + j * 2 + 0] = scale * q0 - min;
      dst[i * QK_K + j * 2 + 1] = scale * q1 - min;
    }
  }
}

// Dequantize Q3_K block (super-block)
// Block structure: QK_K/8 bytes hmask, QK_K/4 bytes qs, 12 bytes scales, fp16 d
inline void dequantize_block_q3_K(const uint8_t* src, float* dst, int64_t k) {
  const int nb = k / QK_K;

  for (int i = 0; i < nb; ++i) {
    const uint8_t* hmask = src;
    src += QK_K / 8;

    const uint8_t* qs = src;
    src += QK_K / 4;

    const uint8_t* scales = src;
    src += 12;

    const uint16_t* d_ptr = reinterpret_cast<const uint16_t*>(src);
    float d = fp16_to_fp32(*d_ptr);
    src += 2;

    // Decode scales (6-bit values packed into 12 bytes for 16 scales)
    int8_t sc[16];
    for (int j = 0; j < 16; ++j) {
      int idx = j / 2;
      int shift = (j % 2) * 4;
      int val = (scales[idx] >> shift) & 0x0F;

      // Get high bits from scales[8..11]
      int high_idx = 8 + j / 4;
      int high_shift = (j % 4) * 2;
      val |= ((scales[high_idx] >> high_shift) & 0x03) << 4;

      sc[j] = static_cast<int8_t>(val - 32);
    }

    // Dequantize (3-bit = 2 low bits from qs + 1 high bit from hmask)
    for (int j = 0; j < QK_K; j += 2) {
      int scale_idx = j / 16;
      float scale = d * sc[scale_idx];

      int byte_idx = j / 4;
      int bit_pos = (j % 4) * 2;

      int q0 = (qs[byte_idx] >> bit_pos) & 0x03;
      int q1 = (qs[byte_idx] >> (bit_pos + 2)) & 0x03;

      // Add high bit from hmask
      int hmask_idx = j / 8;
      int hmask_bit0 = (hmask[hmask_idx] >> (j % 8)) & 1;
      int hmask_bit1 = (hmask[hmask_idx] >> ((j + 1) % 8)) & 1;

      q0 |= hmask_bit0 << 2;
      q1 |= hmask_bit1 << 2;

      // Center around 0
      q0 -= 4;
      q1 -= 4;

      dst[i * QK_K + j] = scale * q0;
      dst[i * QK_K + j + 1] = scale * q1;
    }
  }
}

// Dequantize Q2_K block (super-block)
// Block structure: QK_K/16 bytes scales, QK_K/4 bytes qs, fp16 d, fp16 dmin
inline void dequantize_block_q2_K(const uint8_t* src, float* dst, int64_t k) {
  const int nb = k / QK_K;

  for (int i = 0; i < nb; ++i) {
    const uint8_t* scales = src;
    src += QK_K / 16;

    const uint8_t* qs = src;
    src += QK_K / 4;

    const uint16_t* dm = reinterpret_cast<const uint16_t*>(src);
    float d = fp16_to_fp32(dm[0]);
    float dmin = fp16_to_fp32(dm[1]);
    src += 4;

    // Dequantize (2-bit quants)
    for (int j = 0; j < QK_K; ++j) {
      int scale_idx = j / 16;
      uint8_t sc = scales[scale_idx];
      float scale = d * (sc & 0x0F);
      float min = dmin * (sc >> 4);

      int byte_idx = j / 4;
      int bit_pos = (j % 4) * 2;
      int q = (qs[byte_idx] >> bit_pos) & 0x03;

      dst[i * QK_K + j] = scale * q - min;
    }
  }
}

// MXFP4 lookup table (E2M1 format, doubled values)
// Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
static constexpr int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

// Convert E8M0 (8-bit exponent, 0 mantissa) to float32, divided by 2
// Used for MXFP4 since the E2M1 values in the lookup table are doubled
inline float e8m0_to_fp32_half(uint8_t x) {
  uint32_t bits;

  if (x < 2) {
    // Denormalized: 2^(-128), 2^(-127)
    bits = 0x00200000 << x;
  } else {
    // Normalized: 0.5 * 2^(x-127) = 2^(x-128)
    bits = static_cast<uint32_t>(x - 1) << 23;
  }

  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}

// Dequantize MXFP4 block
// Block structure: 1 byte E8M0 exponent, 16 bytes of 4-bit quants (32 values)
inline void dequantize_block_mxfp4(const uint8_t* src, float* dst, int64_t k) {
  const int nb = k / QK_MXFP4;

  for (int i = 0; i < nb; ++i) {
    const uint8_t e = src[0];
    const uint8_t* qs = src + 1;
    src += 17;  // 1 byte exponent + 16 bytes quants

    const float scale = e8m0_to_fp32_half(e);

    for (int j = 0; j < QK_MXFP4 / 2; ++j) {
      const int8_t v0 = kvalues_mxfp4[qs[j] & 0x0F];  // Lower 4 bits
      const int8_t v1 = kvalues_mxfp4[qs[j] >> 4];    // Upper 4 bits

      dst[i * QK_MXFP4 + j] = v0 * scale;
      dst[i * QK_MXFP4 + j + QK_MXFP4 / 2] = v1 * scale;
    }
  }
}

// Convert fp32 to fp16 (uint16_t)
inline uint16_t fp32_to_fp16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(f));

  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = (bits >> 13) & 0x3FF;

  if (exponent <= 0) {
    // Underflow -> zero or subnormal
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);  // Zero
    }
    // Subnormal
    mantissa |= 0x400;  // Add implicit 1
    mantissa >>= (1 - exponent);
    return static_cast<uint16_t>(sign | mantissa);
  } else if (exponent >= 31) {
    // Overflow -> infinity
    return static_cast<uint16_t>(sign | 0x7C00);
  } else {
    // Normal number
    return static_cast<uint16_t>(sign | (exponent << 10) | mantissa);
  }
}

// Dequantize MXFP4 to float16 (instead of float32)
inline bool dequantize_mxfp4_to_fp16(const uint8_t* src, uint16_t* dst, int64_t k) {
  const int nb = k / QK_MXFP4;

  for (int i = 0; i < nb; ++i) {
    const uint8_t e = src[0];
    const uint8_t* qs = src + 1;
    src += 17;  // 1 byte exponent + 16 bytes quants

    const float scale = e8m0_to_fp32_half(e);

    for (int j = 0; j < QK_MXFP4 / 2; ++j) {
      const int8_t v0 = kvalues_mxfp4[qs[j] & 0x0F];
      const int8_t v1 = kvalues_mxfp4[qs[j] >> 4];

      // Convert float32 to float16
      dst[i * QK_MXFP4 + j] = fp32_to_fp16(v0 * scale);
      dst[i * QK_MXFP4 + j + QK_MXFP4 / 2] = fp32_to_fp16(v1 * scale);
    }
  }

  return true;
}

// Dequantize F16 to F32
inline void dequantize_f16(const uint8_t* src, float* dst, int64_t k) {
  const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
  for (int64_t i = 0; i < k; ++i) {
    dst[i] = fp16_to_fp32(src16[i]);
  }
}

// Main dequantization function
inline bool dequantize_tensor(const uint8_t* src, float* dst, int64_t numel, ggml_type type) {
  switch (type) {
    case ggml_type::F32:
      std::memcpy(dst, src, numel * sizeof(float));
      return true;

    case ggml_type::F16:
      dequantize_f16(src, dst, numel);
      return true;

    case ggml_type::Q4_0:
      dequantize_block_q4_0(src, dst, numel);
      return true;

    case ggml_type::Q5_0:
      dequantize_block_q5_0(src, dst, numel);
      return true;

    case ggml_type::Q8_0:
      dequantize_block_q8_0(src, dst, numel);
      return true;

    case ggml_type::Q2_K:
      dequantize_block_q2_K(src, dst, numel);
      return true;

    case ggml_type::Q3_K:
      dequantize_block_q3_K(src, dst, numel);
      return true;

    case ggml_type::Q4_K:
      dequantize_block_q4_K(src, dst, numel);
      return true;

    case ggml_type::Q5_K:
      dequantize_block_q5_K(src, dst, numel);
      return true;

    case ggml_type::Q6_K:
      dequantize_block_q6_K(src, dst, numel);
      return true;

    case ggml_type::MXFP4:
      dequantize_block_mxfp4(src, dst, numel);
      return true;

    default:
      // Unsupported types (e.g., IQ*) will return false and show a warning
      return false;
  }
}

// Get number of elements per block for quantized type
inline int64_t get_block_size(ggml_type type) {
  switch (type) {
    case ggml_type::F32:
    case ggml_type::F16:
      return 1;
    case ggml_type::Q4_0:
    case ggml_type::Q4_1:
    case ggml_type::Q5_0:
    case ggml_type::Q5_1:
    case ggml_type::Q8_0:
    case ggml_type::Q8_1:
    case ggml_type::MXFP4:
      return 32;
    case ggml_type::Q2_K:
    case ggml_type::Q3_K:
    case ggml_type::Q4_K:
    case ggml_type::Q5_K:
    case ggml_type::Q6_K:
    case ggml_type::Q8_K:
      return 256;
    default:
      return 1;
  }
}

}  // namespace coalsack
