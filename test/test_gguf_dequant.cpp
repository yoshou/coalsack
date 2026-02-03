#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "gguf_dequant.h"

using namespace coalsack;

// Test fp16 <-> fp32 conversion
TEST(GgufDequantTest, Fp16ToFp32Basic) {
  // Test 1.0
  EXPECT_FLOAT_EQ(fp16_to_fp32(0x3C00), 1.0f);
  
  // Test 0.0
  EXPECT_FLOAT_EQ(fp16_to_fp32(0x0000), 0.0f);
  
  // Test -1.0
  EXPECT_FLOAT_EQ(fp16_to_fp32(0xBC00), -1.0f);
  
  // Test 2.0
  EXPECT_FLOAT_EQ(fp16_to_fp32(0x4000), 2.0f);
  
  // Test 0.5
  EXPECT_FLOAT_EQ(fp16_to_fp32(0x3800), 0.5f);
}

TEST(GgufDequantTest, Fp32ToFp16Basic) {
  EXPECT_EQ(fp32_to_fp16(1.0f), 0x3C00);
  EXPECT_EQ(fp32_to_fp16(0.0f), 0x0000);
  EXPECT_EQ(fp32_to_fp16(-1.0f), 0xBC00);
  EXPECT_EQ(fp32_to_fp16(2.0f), 0x4000);
  EXPECT_EQ(fp32_to_fp16(0.5f), 0x3800);
}

TEST(GgufDequantTest, Fp16RoundTrip) {
  std::vector<float> test_values = {
      0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 0.25f, 
      3.14159f, -3.14159f, 100.0f, -100.0f
  };
  
  for (float val : test_values) {
    uint16_t fp16 = fp32_to_fp16(val);
    float converted = fp16_to_fp32(fp16);
    // Allow for precision loss in fp16
    EXPECT_NEAR(converted, val, std::abs(val) * 0.001f + 0.0001f) 
        << "Failed for value: " << val;
  }
}

// Test Q4_0 dequantization
TEST(GgufDequantTest, DequantizeQ4_0) {
  const int64_t n = QK4_0;  // 32 elements
  std::vector<uint8_t> src(2 + QK4_0 / 2);  // fp16 scale + 16 bytes of quants
  std::vector<float> dst(n);
  
  // Set scale to 1.0 (fp16)
  uint16_t* scale_ptr = reinterpret_cast<uint16_t*>(src.data());
  *scale_ptr = fp32_to_fp16(1.0f);
  
  // Set all quantized values to 0 (which maps to -8 after offset)
  for (int i = 0; i < QK4_0 / 2; ++i) {
    src[2 + i] = 0x00;  // Both nibbles are 0
  }
  
  dequantize_block_q4_0(src.data(), dst.data(), n);
  
  // All values should be -8.0
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(dst[i], -8.0f) << "Failed at index: " << i;
  }
}

TEST(GgufDequantTest, DequantizeQ4_0_Scale) {
  const int64_t n = QK4_0;
  std::vector<uint8_t> src(2 + QK4_0 / 2);
  std::vector<float> dst(n);
  
  // Set scale to 2.0
  uint16_t* scale_ptr = reinterpret_cast<uint16_t*>(src.data());
  *scale_ptr = fp32_to_fp16(2.0f);
  
  // Set quantized values: first nibble=8 (maps to 0), second nibble=9 (maps to 1)
  src[2] = 0x98;  // First byte: low=8, high=9
  
  dequantize_block_q4_0(src.data(), dst.data(), n);
  
  // First value: (8-8) * 2.0 = 0.0
  EXPECT_FLOAT_EQ(dst[0], 0.0f);
  // Second value: (9-8) * 2.0 = 2.0
  EXPECT_FLOAT_EQ(dst[1], 2.0f);
}

// Test Q8_0 dequantization
TEST(GgufDequantTest, DequantizeQ8_0) {
  const int64_t n = QK8_0;  // 32 elements
  std::vector<uint8_t> src(2 + QK8_0);  // fp16 scale + 32 bytes of int8
  std::vector<float> dst(n);
  
  // Set scale to 0.5
  uint16_t* scale_ptr = reinterpret_cast<uint16_t*>(src.data());
  *scale_ptr = fp32_to_fp16(0.5f);
  
  // Set quantized values
  int8_t* quants = reinterpret_cast<int8_t*>(src.data() + 2);
  for (int i = 0; i < QK8_0; ++i) {
    quants[i] = static_cast<int8_t>(i - 16);
  }
  
  dequantize_block_q8_0(src.data(), dst.data(), n);
  
  // Check values
  for (int i = 0; i < n; ++i) {
    float expected = (i - 16) * 0.5f;
    EXPECT_FLOAT_EQ(dst[i], expected) << "Failed at index: " << i;
  }
}

// Test F16 dequantization
TEST(GgufDequantTest, DequantizeF16) {
  const int64_t n = 10;
  std::vector<uint16_t> src_fp16(n);
  std::vector<float> dst(n);
  
  // Create test data in fp16
  for (int i = 0; i < n; ++i) {
    src_fp16[i] = fp32_to_fp16(static_cast<float>(i) * 0.5f);
  }
  
  dequantize_f16(reinterpret_cast<uint8_t*>(src_fp16.data()), dst.data(), n);
  
  // Verify
  for (int i = 0; i < n; ++i) {
    float expected = static_cast<float>(i) * 0.5f;
    EXPECT_NEAR(dst[i], expected, 0.001f) << "Failed at index: " << i;
  }
}

// Test dequantize_tensor with F32
TEST(GgufDequantTest, DequantizeTensorF32) {
  const int64_t n = 10;
  std::vector<float> src(n);
  std::vector<float> dst(n);
  
  for (int i = 0; i < n; ++i) {
    src[i] = static_cast<float>(i) * 1.5f;
  }
  
  bool result = dequantize_tensor(
      reinterpret_cast<uint8_t*>(src.data()), 
      dst.data(), 
      n, 
      ggml_type::F32
  );
  
  EXPECT_TRUE(result);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(dst[i], src[i]) << "Failed at index: " << i;
  }
}

// Test dequantize_tensor with F16
TEST(GgufDequantTest, DequantizeTensorF16) {
  const int64_t n = 10;
  std::vector<uint16_t> src(n);
  std::vector<float> dst(n);
  
  for (int i = 0; i < n; ++i) {
    src[i] = fp32_to_fp16(static_cast<float>(i) * 0.5f);
  }
  
  bool result = dequantize_tensor(
      reinterpret_cast<uint8_t*>(src.data()), 
      dst.data(), 
      n, 
      ggml_type::F16
  );
  
  EXPECT_TRUE(result);
  for (int i = 0; i < n; ++i) {
    float expected = static_cast<float>(i) * 0.5f;
    EXPECT_NEAR(dst[i], expected, 0.001f) << "Failed at index: " << i;
  }
}

// Test dequantize_tensor with Q4_0
TEST(GgufDequantTest, DequantizeTensorQ4_0) {
  const int64_t n = QK4_0;
  std::vector<uint8_t> src(2 + QK4_0 / 2);
  std::vector<float> dst(n);
  
  uint16_t* scale_ptr = reinterpret_cast<uint16_t*>(src.data());
  *scale_ptr = fp32_to_fp16(1.0f);
  
  for (int i = 0; i < QK4_0 / 2; ++i) {
    src[2 + i] = 0x88;  // Both nibbles are 8 (maps to 0 after offset)
  }
  
  bool result = dequantize_tensor(src.data(), dst.data(), n, ggml_type::Q4_0);
  
  EXPECT_TRUE(result);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(dst[i], 0.0f) << "Failed at index: " << i;
  }
}

// Test get_block_size
TEST(GgufDequantTest, GetBlockSize) {
  EXPECT_EQ(get_block_size(ggml_type::F32), 1);
  EXPECT_EQ(get_block_size(ggml_type::F16), 1);
  EXPECT_EQ(get_block_size(ggml_type::Q4_0), 32);
  EXPECT_EQ(get_block_size(ggml_type::Q4_1), 32);
  EXPECT_EQ(get_block_size(ggml_type::Q5_0), 32);
  EXPECT_EQ(get_block_size(ggml_type::Q5_1), 32);
  EXPECT_EQ(get_block_size(ggml_type::Q8_0), 32);
  EXPECT_EQ(get_block_size(ggml_type::Q8_1), 32);
  EXPECT_EQ(get_block_size(ggml_type::Q2_K), 256);
  EXPECT_EQ(get_block_size(ggml_type::Q3_K), 256);
  EXPECT_EQ(get_block_size(ggml_type::Q4_K), 256);
  EXPECT_EQ(get_block_size(ggml_type::Q5_K), 256);
  EXPECT_EQ(get_block_size(ggml_type::Q6_K), 256);
  EXPECT_EQ(get_block_size(ggml_type::MXFP4), 32);
}

// Test Q5_0 dequantization
TEST(GgufDequantTest, DequantizeQ5_0Basic) {
  const int64_t n = QK5_0;
  std::vector<uint8_t> src(2 + 4 + QK5_0 / 2);  // scale + qh + qs
  std::vector<float> dst(n);
  
  // Set scale to 1.0
  uint16_t* scale_ptr = reinterpret_cast<uint16_t*>(src.data());
  *scale_ptr = fp32_to_fp16(1.0f);
  
  // Set all high bits to 0
  std::memset(src.data() + 2, 0, 4);
  
  // Set all low bits to 16 (10000 in binary, after adding high bit 0)
  // This should map to 0 after subtracting 16
  for (int i = 0; i < QK5_0 / 2; ++i) {
    src[2 + 4 + i] = 0x00;  // 0x00 means both nibbles are 0, which maps to -16 before adding high bit
  }
  
  dequantize_block_q5_0(src.data(), dst.data(), n);
  
  // All values should be -16.0 (since 0 - 16 = -16)
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(dst[i], -16.0f) << "Failed at index: " << i;
  }
}

// Test MXFP4 basic dequantization
TEST(GgufDequantTest, DequantizeMXFP4Basic) {
  const int64_t n = QK_MXFP4;  // 32 elements
  std::vector<uint8_t> src(17);  // 1 byte exponent + 16 bytes quants
  std::vector<float> dst(n);
  
  // Set exponent to 128 (scale = 1.0)
  src[0] = 128;
  
  // Set all quants to 0 (maps to 0 in lookup table)
  for (int i = 1; i < 17; ++i) {
    src[i] = 0x00;  // Both nibbles are 0
  }
  
  dequantize_block_mxfp4(src.data(), dst.data(), n);
  
  // All values should be 0.0
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(dst[i], 0.0f) << "Failed at index: " << i;
  }
}

// Test e8m0_to_fp32_half
TEST(GgufDequantTest, E8M0ToFp32Half) {
  // Test e8m0_to_fp32_half with a few values
  // e8m0 = 127 should give 2^(127-128) = 2^(-1) = 0.5
  float result = e8m0_to_fp32_half(127);
  EXPECT_FLOAT_EQ(result, 0.5f);
  
  // e8m0 = 128 should give 2^(128-128) = 2^0 = 1.0
  result = e8m0_to_fp32_half(128);
  EXPECT_FLOAT_EQ(result, 1.0f);
  
  // e8m0 = 129 should give 2^(129-128) = 2^1 = 2.0
  result = e8m0_to_fp32_half(129);
  EXPECT_FLOAT_EQ(result, 2.0f);
}

// ===== More Complex Tests =====

// Test Q4_0 with multiple blocks and varying values
TEST(GgufDequantTest, DequantizeQ4_0_MultiBlock) {
  const int64_t n = QK4_0 * 3;  // 3 blocks
  std::vector<uint8_t> src((2 + QK4_0 / 2) * 3);
  std::vector<float> dst(n);
  
  // Block 0: scale = 0.5
  uint16_t* scale0 = reinterpret_cast<uint16_t*>(src.data());
  *scale0 = fp32_to_fp16(0.5f);
  for (int i = 0; i < QK4_0 / 2; ++i) {
    src[2 + i] = (i % 16) | ((i % 16) << 4);  // Varying values
  }
  
  // Block 1: scale = 1.5
  uint16_t* scale1 = reinterpret_cast<uint16_t*>(src.data() + (2 + QK4_0 / 2));
  *scale1 = fp32_to_fp16(1.5f);
  for (int i = 0; i < QK4_0 / 2; ++i) {
    src[2 + QK4_0 / 2 + 2 + i] = 0xFF;  // Both nibbles max (15)
  }
  
  // Block 2: scale = 0.25
  uint16_t* scale2 = reinterpret_cast<uint16_t*>(src.data() + (2 + QK4_0 / 2) * 2);
  *scale2 = fp32_to_fp16(0.25f);
  for (int i = 0; i < QK4_0 / 2; ++i) {
    src[(2 + QK4_0 / 2) * 2 + 2 + i] = 0x88;  // Both nibbles 8 (zero after offset)
  }
  
  dequantize_block_q4_0(src.data(), dst.data(), n);
  
  // Verify block 0
  for (int i = 0; i < 16; ++i) {
    int q = i % 16;
    float expected = (q - 8) * 0.5f;
    EXPECT_FLOAT_EQ(dst[i * 2], expected);
    EXPECT_FLOAT_EQ(dst[i * 2 + 1], expected);
  }
  
  // Verify block 1: 0xF = 15, 15-8 = 7, scale=1.5, result=10.5
  for (int i = 0; i < QK4_0; ++i) {
    EXPECT_NEAR(dst[QK4_0 + i], 7.0f * 1.5f, 0.1f);
  }
  
  // Verify block 2: 0x8 = 8, 8-8 = 0
  for (int i = 0; i < QK4_0; ++i) {
    EXPECT_NEAR(dst[QK4_0 * 2 + i], 0.0f, 0.1f);
  }
}

// Test Q5_0 with complex bit patterns
TEST(GgufDequantTest, DequantizeQ5_0_Complex) {
  const int64_t n = QK5_0;
  std::vector<uint8_t> src(2 + 4 + QK5_0 / 2);
  std::vector<float> dst(n);
  
  // Set scale to 0.3
  uint16_t* scale_ptr = reinterpret_cast<uint16_t*>(src.data());
  *scale_ptr = fp32_to_fp16(0.3f);
  
  // Set high bits pattern
  uint32_t qh_pattern = 0xAAAA5555;  // Alternating pattern
  std::memcpy(src.data() + 2, &qh_pattern, 4);
  
  // Set low bits to incrementing pattern
  for (int i = 0; i < QK5_0 / 2; ++i) {
    src[2 + 4 + i] = (i & 0x0F) | ((i & 0x0F) << 4);
  }
  
  dequantize_block_q5_0(src.data(), dst.data(), n);
  
  // Verify first few values
  for (int i = 0; i < 4; ++i) {
    // Each value has specific high bit and low bits
    EXPECT_TRUE(std::isfinite(dst[i]));
  }
}

// Test Q8_0 with realistic distribution
TEST(GgufDequantTest, DequantizeQ8_0_Distribution) {
  const int64_t n = QK8_0 * 2;  // 2 blocks
  std::vector<uint8_t> src((2 + QK8_0) * 2);
  std::vector<float> dst(n);
  
  // Block 0: scale = 0.05, values from -128 to 127
  uint16_t* scale0 = reinterpret_cast<uint16_t*>(src.data());
  *scale0 = fp32_to_fp16(0.05f);
  int8_t* quants0 = reinterpret_cast<int8_t*>(src.data() + 2);
  for (int i = 0; i < QK8_0; ++i) {
    quants0[i] = static_cast<int8_t>(i * 8 - 128);
  }
  
  // Block 1: scale = 0.02, alternating positive/negative
  uint16_t* scale1 = reinterpret_cast<uint16_t*>(src.data() + 2 + QK8_0);
  *scale1 = fp32_to_fp16(0.02f);
  int8_t* quants1 = reinterpret_cast<int8_t*>(src.data() + 2 + QK8_0 + 2);
  for (int i = 0; i < QK8_0; ++i) {
    quants1[i] = (i % 2 == 0) ? 100 : -100;
  }
  
  dequantize_block_q8_0(src.data(), dst.data(), n);
  
  // Verify block 0 (allow fp16 precision loss)
  for (int i = 0; i < QK8_0; ++i) {
    float expected = (i * 8 - 128) * 0.05f;
    EXPECT_NEAR(dst[i], expected, 0.005f);
  }
  
  // Verify block 1 (allow fp16 precision loss)
  for (int i = 0; i < QK8_0; ++i) {
    float expected = ((i % 2 == 0) ? 100 : -100) * 0.02f;
    EXPECT_NEAR(dst[QK8_0 + i], expected, 0.005f);
  }
}

// Test Q4_K with realistic scale patterns
TEST(GgufDequantTest, DequantizeQ4_K_Complex) {
  const int64_t n = QK_K;  // 256 elements
  std::vector<uint8_t> src(4 + K_SCALE_SIZE + QK_K / 2);
  std::vector<float> dst(n);
  
  // Set d and min
  uint16_t* dm = reinterpret_cast<uint16_t*>(src.data());
  dm[0] = fp32_to_fp16(0.1f);  // d
  dm[1] = fp32_to_fp16(0.05f); // min
  
  // Set scales: Carefully craft a pattern that tests scale decoding
  // This pattern exercises both j<4 and j>=4 paths in get_scale_min_k4
  uint8_t* scales = src.data() + 4;
  // First 4 bytes: scales for sub-blocks 0-3 (lower 6 bits)
  scales[0] = 0x3F;  // 63
  scales[1] = 0x80;  // High bits set (tests >> 6)
  scales[2] = 0xC5;  // Mixed bits
  scales[3] = 0x2A;  // 42
  // Next 4 bytes: mins for sub-blocks 0-3 (lower 6 bits)
  scales[4] = 0x15;  // 21
  scales[5] = 0xF0;  // High bits set
  scales[6] = 0x88;  // Mixed bits
  scales[7] = 0x1F;  // 31
  // Last 4 bytes: high bits for scales 4-7
  scales[8] = 0xAA;  // 10101010
  scales[9] = 0x55;  // 01010101
  scales[10] = 0xCC; // 11001100
  scales[11] = 0x33; // 00110011
  
  // Set quants: Different patterns for different sub-blocks
  uint8_t* qs = src.data() + 4 + K_SCALE_SIZE;
  for (int i = 0; i < QK_K / 2; ++i) {
    // Create distinct patterns for each 32-byte section
    uint8_t val = ((i / 32) * 3 + (i % 32)) % 16;
    qs[i] = val | ((15 - val) << 4);
  }
  
  dequantize_block_q4_K(src.data(), dst.data(), n);
  
  // Verify specific expected patterns
  // The decoding should produce different results for different sub-blocks
  // based on the scale/min values
  std::vector<float> sub_block_means(8);
  for (int sb = 0; sb < 8; ++sb) {
    float sum = 0;
    for (int i = 0; i < 32; ++i) {
      float val = dst[sb * 32 + i];
      EXPECT_TRUE(std::isfinite(val)) << "Non-finite at index: " << (sb * 32 + i);
      sum += val;
    }
    sub_block_means[sb] = sum / 32.0f;
  }
  
  // Each sub-block should have different mean due to different scales
  for (int i = 1; i < 8; ++i) {
    // Not all means should be identical (would indicate scale decoding bug)
    bool all_same = true;
    for (int j = 0; j < i; ++j) {
      if (std::abs(sub_block_means[i] - sub_block_means[j]) > 0.001f) {
        all_same = false;
        break;
      }
    }
    if (i >= 2) {
      EXPECT_FALSE(all_same) << "Sub-blocks have identical means, scale decoding may be incorrect";
    }
  }
}

// Test Q5_K with high bit variations
TEST(GgufDequantTest, DequantizeQ5_K_Complex) {
  const int64_t n = QK_K;
  std::vector<uint8_t> src(4 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2);
  std::vector<float> dst(n);
  
  // Set d and dmin
  uint16_t* dm = reinterpret_cast<uint16_t*>(src.data());
  dm[0] = fp32_to_fp16(0.08f);
  dm[1] = fp32_to_fp16(0.03f);
  
  // Set scales: same pattern as Q4_K to test scale decoding
  uint8_t* scales = src.data() + 4;
  scales[0] = 0x3F;
  scales[1] = 0x80;
  scales[2] = 0xC5;
  scales[3] = 0x2A;
  scales[4] = 0x15;
  scales[5] = 0xF0;
  scales[6] = 0x88;
  scales[7] = 0x1F;
  scales[8] = 0xAA;
  scales[9] = 0x55;
  scales[10] = 0xCC;
  scales[11] = 0x33;
  
  // Set high bits with specific pattern to test bit extraction
  uint8_t* qh = src.data() + 4 + K_SCALE_SIZE;
  for (int i = 0; i < QK_K / 8; ++i) {
    // Create pattern that ensures different elements get different high bits
    qh[i] = ((i % 4) << 6) | ((i % 8) << 3) | (i % 8);
  }
  
  // Set low bits with distinct pattern
  uint8_t* qs = src.data() + 4 + K_SCALE_SIZE + QK_K / 8;
  for (int i = 0; i < QK_K / 2; ++i) {
    qs[i] = (i % 256);
  }
  
  dequantize_block_q5_K(src.data(), dst.data(), n);
  
  // Verify output and check that high bits have an effect
  int high_bit_variations = 0;
  for (int i = 0; i < n - 1; ++i) {
    EXPECT_TRUE(std::isfinite(dst[i])) << "Non-finite at index: " << i;
    // Count significant differences (high bit should cause ~16x difference in quant value)
    if (std::abs(dst[i] - dst[i + 1]) > 0.1f) {
      high_bit_variations++;
    }
  }
  // Should have many variations if high bits are being used correctly
  EXPECT_GT(high_bit_variations, 50) << "Not enough variations, high bits may not be decoded correctly";
}

// Test Q6_K with full range
TEST(GgufDequantTest, DequantizeQ6_K_Complex) {
  const int64_t n = QK_K;
  std::vector<uint8_t> src(QK_K / 2 + QK_K / 4 + QK_K / 16 + 2);
  std::vector<float> dst(n);
  
  // Set ql (low bits)
  uint8_t* ql = src.data();
  for (int i = 0; i < QK_K / 2; ++i) {
    ql[i] = (i * 13) % 256;
  }
  
  // Set qh (high bits)
  uint8_t* qh = src.data() + QK_K / 2;
  for (int i = 0; i < QK_K / 4; ++i) {
    qh[i] = (i * 31) % 256;
  }
  
  // Set scales
  int8_t* scales = reinterpret_cast<int8_t*>(src.data() + QK_K / 2 + QK_K / 4);
  for (int i = 0; i < QK_K / 16; ++i) {
    scales[i] = static_cast<int8_t>((i * 11) % 128 - 64);
  }
  
  // Set d
  uint16_t* d_ptr = reinterpret_cast<uint16_t*>(src.data() + QK_K / 2 + QK_K / 4 + QK_K / 16);
  *d_ptr = fp32_to_fp16(0.07f);
  
  dequantize_block_q6_K(src.data(), dst.data(), n);
  
  // Verify output
  for (int i = 0; i < n; ++i) {
    EXPECT_TRUE(std::isfinite(dst[i])) << "Non-finite at index: " << i;
  }
}

// Test Q3_K with 3-bit patterns
TEST(GgufDequantTest, DequantizeQ3_K_Complex) {
  const int64_t n = QK_K;
  std::vector<uint8_t> src(QK_K / 8 + QK_K / 4 + 12 + 2);
  std::vector<float> dst(n);
  
  // Set hmask
  uint8_t* hmask = src.data();
  for (int i = 0; i < QK_K / 8; ++i) {
    hmask[i] = (i % 4 == 0) ? 0xF0 : ((i % 4 == 1) ? 0x0F : 0xAA);
  }
  
  // Set qs
  uint8_t* qs = src.data() + QK_K / 8;
  for (int i = 0; i < QK_K / 4; ++i) {
    qs[i] = (i * 23) % 256;
  }
  
  // Set scales
  uint8_t* scales = src.data() + QK_K / 8 + QK_K / 4;
  for (int i = 0; i < 12; ++i) {
    scales[i] = (i * 19 + 32) % 256;
  }
  
  // Set d
  uint16_t* d_ptr = reinterpret_cast<uint16_t*>(src.data() + QK_K / 8 + QK_K / 4 + 12);
  *d_ptr = fp32_to_fp16(0.12f);
  
  dequantize_block_q3_K(src.data(), dst.data(), n);
  
  // Verify output
  for (int i = 0; i < n; ++i) {
    EXPECT_TRUE(std::isfinite(dst[i])) << "Non-finite at index: " << i;
  }
}

// Test Q2_K with 2-bit patterns
TEST(GgufDequantTest, DequantizeQ2_K_Complex) {
  const int64_t n = QK_K;
  std::vector<uint8_t> src(QK_K / 16 + QK_K / 4 + 4);
  std::vector<float> dst(n);
  
  // Set scales
  uint8_t* scales = src.data();
  for (int i = 0; i < QK_K / 16; ++i) {
    scales[i] = (i * 29 + 64) % 256;
  }
  
  // Set qs
  uint8_t* qs = src.data() + QK_K / 16;
  for (int i = 0; i < QK_K / 4; ++i) {
    qs[i] = (i % 4) * 0x55;  // Pattern: 00, 01, 10, 11
  }
  
  // Set d and dmin
  uint16_t* dm = reinterpret_cast<uint16_t*>(src.data() + QK_K / 16 + QK_K / 4);
  dm[0] = fp32_to_fp16(0.15f);
  dm[1] = fp32_to_fp16(0.08f);
  
  dequantize_block_q2_K(src.data(), dst.data(), n);
  
  // Verify output
  for (int i = 0; i < n; ++i) {
    EXPECT_TRUE(std::isfinite(dst[i])) << "Non-finite at index: " << i;
  }
}

// Test fp16 special values
TEST(GgufDequantTest, Fp16SpecialValues) {
  // Test infinity
  uint16_t pos_inf = 0x7C00;
  EXPECT_TRUE(std::isinf(fp16_to_fp32(pos_inf)));
  EXPECT_GT(fp16_to_fp32(pos_inf), 0);
  
  uint16_t neg_inf = 0xFC00;
  EXPECT_TRUE(std::isinf(fp16_to_fp32(neg_inf)));
  EXPECT_LT(fp16_to_fp32(neg_inf), 0);
  
  // Test NaN
  uint16_t nan_val = 0x7C01;
  EXPECT_TRUE(std::isnan(fp16_to_fp32(nan_val)));
  
  // Test subnormal numbers
  uint16_t subnormal = 0x0001;  // Smallest positive subnormal
  float result = fp16_to_fp32(subnormal);
  EXPECT_GT(result, 0);
  EXPECT_LT(result, 0.0001f);
}

// Test Q5_0 with expected output values to catch implementation errors
TEST(GgufDequantTest, DequantizeQ5_0_ExpectedValues) {
  const int64_t n = QK5_0;
  std::vector<uint8_t> src(2 + 4 + QK5_0 / 2);
  std::vector<float> dst(n);
  
  // Set d = 1.0
  uint16_t* d_ptr = reinterpret_cast<uint16_t*>(src.data());
  *d_ptr = fp32_to_fp16(1.0f);
  
  // Set qh = 0x00000000 (no high bits set)
  uint32_t qh_bits = 0x00000000;
  std::memcpy(src.data() + 2, &qh_bits, 4);
  
  // Set qs with distinctive pattern: first byte = 0x21 (low nibble=1, high nibble=2)
  uint8_t* qs = src.data() + 6;
  qs[0] = 0x21;  // Low nibble: 1, High nibble: 2
  for (int i = 1; i < QK5_0 / 2; ++i) {
    qs[i] = 0x00;
  }
  
  dequantize_block_q5_0(src.data(), dst.data(), n);
  
  // Correct implementation (non-interleaved output):
  // j=0: x0 = (qs[0] & 0x0F) | 0 = 1, centered: 1-16 = -15 -> dst[0] = 1.0 * -15 = -15.0
  //      x1 = (qs[0] >> 4) | 0 = 2, centered: 2-16 = -14 -> dst[16] = 1.0 * -14 = -14.0
  // j=1: x0 = (qs[1] & 0x0F) | 0 = 0, centered: 0-16 = -16 -> dst[1] = 1.0 * -16 = -16.0
  //
  // Wrong implementation (interleaved output):
  // j=0: x0 -> dst[0], x1 -> dst[1] (should be dst[16])
  // So dst[1] would be -14.0, dst[16] would be -16.0
  
  EXPECT_NEAR(dst[0], -15.0f, 0.01f);   // First element from low nibble of qs[0]
  EXPECT_NEAR(dst[16], -14.0f, 0.01f);  // First element of second half from high nibble of qs[0]
  EXPECT_NEAR(dst[1], -16.0f, 0.01f);   // Second element from low nibble of qs[1]
}

// Test Q4_K with known scale decoding to catch implementation errors
// This test is designed to fail with the current wrong implementation
TEST(GgufDequantTest, DequantizeQ4_K_ScaleDecoding) {
  const int64_t n = QK_K;
  std::vector<uint8_t> src(4 + K_SCALE_SIZE + QK_K / 2);
  std::vector<float> dst(n);
  
  // Set d and min
  uint16_t* dm = reinterpret_cast<uint16_t*>(src.data());
  dm[0] = fp32_to_fp16(1.0f);
  dm[1] = fp32_to_fp16(0.0f);
  
  // Set scales - use pattern that exposes wrong block_idx = j / 16 logic
  uint8_t* scales = src.data() + 4;
  scales[0] = 1;   // scale[0] = 1
  scales[1] = 2;   // scale[1] = 2
  scales[2] = 4;   // scale[2] = 4
  scales[3] = 8;   // scale[3] = 8
  for (int i = 4; i < 12; ++i) scales[i] = 0;
  
  // Set quants with ascending pattern to detect reading order issues
  uint8_t* qs = src.data() + 4 + K_SCALE_SIZE;
  for (int i = 0; i < QK_K / 2; ++i) {
    // Create unique pattern: use i as the value
    uint8_t low = (i % 16);
    uint8_t high = ((i + 1) % 16);
    qs[i] = (high << 4) | low;
  }
  
  dequantize_block_q4_K(src.data(), dst.data(), n);
  
  // The correct implementation (get_scale_min_k4 based) produces a very specific pattern
  // The wrong implementation (block_idx = j / 16) produces a different pattern
  
  // With correct implementation:
  // First 32 bytes of qs (indices 0-31) are split into 64 output values:
  //   - qs[0-31] low nibbles (values 0-15, then 0-15) -> dst[0-31] with scale[0]=1
  //   - qs[0-31] high nibbles (values 1-16, then 1-16 循環) -> dst[32-63] with scale[1]=2
  
  // With wrong implementation:
  // qs[0] = 0x10 -> q0=0, q1=1 -> dst[0]=1*0=0, dst[1]=1*1=1
  // qs[1] = 0x21 -> q0=1, q1=2 -> dst[2]=1*1=1, dst[3]=1*2=2
  // ...
  // qs[16] changes block_idx to 1 -> scale becomes 2
  
  // Key difference: dst[1] should be from high nibble of qs[0], not low nibble of qs[0]
  // Correct: dst[1] is part of first 32-element block using scale[0]
  // Wrong: dst[1] uses scale[0] but is from high nibble of qs[0]
  
  // Let's check a specific pattern that differs
  // With wrong impl: dst[2] = scale[0] * (qs[1] & 0xF) = 1 * 1 = 1.0
  // With correct impl: dst[2] = scale[0] * (qs[2] & 0xF) = 1 * 2 = 2.0
  
  EXPECT_TRUE(std::isfinite(dst[0]));
  EXPECT_TRUE(std::isfinite(dst[1]));
  EXPECT_TRUE(std::isfinite(dst[2]));
  
  // This specific check should fail with wrong implementation
  // Expected with correct implementation: dst[2] = scale[0] * (qs[2] & 0xF) = 1 * 2 = 2.0
  // Actual with wrong implementation: dst[2] = scale[0] * (qs[1] & 0xF) = 1 * 1 = 1.0
  EXPECT_NEAR(dst[2], 2.0f, 0.1f) << "dst[2] should be 2.0 with correct implementation";
  
  // Another check: dst[32] should use scale[1]=2
  // Expected with correct implementation: dst[32] uses high nibble of qs[0], scale[1]=2
  //   high nibble of qs[0] = 1, so dst[32] = 2 * 1 = 2.0
  // With wrong implementation: dst[32] = scale[1] * (qs[16] & 0xF) = 2 * (16%16) = 0.0
  EXPECT_NEAR(dst[32], 2.0f, 0.1f) << "dst[32] should be 2.0 with correct implementation";
}
