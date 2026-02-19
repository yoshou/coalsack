#include <gtest/gtest.h>

#include "coalsack/gguf/gguf_dequant.h"
#include "coalsack/nn/nn_ops/embedding_lookup_node.h"
#include "coalsack/nn/nn_ops/matmul_mixed_node.h"
#include "coalsack/tensor/dynamic_tensor.h"

using namespace coalsack;

TEST(Float16Test, DtypeSupport) {
  dynamic_tensor t(dtype::float16, {10, 20});
  EXPECT_EQ(t.get_dtype(), dtype::float16);
  EXPECT_EQ(t.bytes(), 10 * 20 * 2);
}

TEST(Float16Test, Fp16ToFp32Conversion) {
  // Test fp16_to_fp32 conversion
  uint16_t fp16_val = 0x3C00;  // 1.0 in fp16
  float fp32_val = fp16_to_fp32(fp16_val);
  EXPECT_FLOAT_EQ(fp32_val, 1.0f);

  // Test fp32_to_fp16 conversion
  uint16_t converted = fp32_to_fp16(1.0f);
  EXPECT_EQ(converted, 0x3C00);
}

TEST(Float16Test, MatMulMixed_Fp32Fp32) {
  // Test fp32 + fp32
  dynamic_tensor a(dtype::float32, {2, 3});
  dynamic_tensor b(dtype::float32, {3, 4});

  // Fill with simple data
  float* a_data = a.data_ptr<float>();
  float* b_data = b.data_ptr<float>();
  for (int i = 0; i < 6; ++i) a_data[i] = static_cast<float>(i + 1);
  for (int i = 0; i < 12; ++i) b_data[i] = static_cast<float>(i + 1);

  matmul_mixed_node matmul;
  dynamic_tensor c = matmul.compute_test(a, b);

  EXPECT_EQ(c.get_dtype(), dtype::float32);
  ASSERT_EQ(c.shape().size(), 2);
  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 4);

  // Verify a simple computation
  float* c_data = c.data_ptr<float>();
  // First element: (1*1 + 2*5 + 3*9) = 1 + 10 + 27 = 38
  EXPECT_FLOAT_EQ(c_data[0], 38.0f);
}

TEST(Float16Test, MatMulMixed_Fp32Fp16) {
  // Test fp32 + fp16 (weight)
  dynamic_tensor a(dtype::float32, {2, 3});
  dynamic_tensor b_fp16(dtype::float16, {3, 4});

  // Fill with simple data
  float* a_data = a.data_ptr<float>();
  uint16_t* b_data = b_fp16.data_ptr<uint16_t>();
  for (int i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i + 1);
  }
  for (int i = 0; i < 12; ++i) {
    b_data[i] = fp32_to_fp16(static_cast<float>(i + 1));
  }

  matmul_mixed_node matmul;
  dynamic_tensor c = matmul.compute_test(a, b_fp16);

  EXPECT_EQ(c.get_dtype(), dtype::float32);
  ASSERT_EQ(c.shape().size(), 2);
  EXPECT_EQ(c.shape()[0], 2);
  EXPECT_EQ(c.shape()[1], 4);

  // Verify the same computation as above
  float* c_data = c.data_ptr<float>();
  EXPECT_NEAR(c_data[0], 38.0f, 0.1f);  // Allow small error due to fp16 conversion
}

TEST(Float16Test, EmbeddingLookup_Fp16) {
  // Test embedding_lookup_node with fp16 weight
  dynamic_tensor ids(dtype::int32, {1, 2});
  ids.data_ptr<int32_t>()[0] = 0;
  ids.data_ptr<int32_t>()[1] = 1;

  dynamic_tensor weight_fp16(dtype::float16, {10, 4});

  // Fill weight with test data
  uint16_t* w_data = weight_fp16.data_ptr<uint16_t>();
  for (int i = 0; i < 10 * 4; ++i) {
    w_data[i] = fp32_to_fp16(static_cast<float>(i));
  }

  embedding_lookup_node emb;
  dynamic_tensor output = emb.compute_test(ids, weight_fp16);

  EXPECT_EQ(output.get_dtype(), dtype::float32);
  ASSERT_EQ(output.shape().size(), 3);
  EXPECT_EQ(output.shape()[0], 1);
  EXPECT_EQ(output.shape()[1], 2);
  EXPECT_EQ(output.shape()[2], 4);

  // Verify embeddings were correctly retrieved and converted
  float* out_data = output.data_ptr<float>();
  // First token (id=0) should have values [0, 1, 2, 3]
  EXPECT_NEAR(out_data[0], 0.0f, 0.01f);
  EXPECT_NEAR(out_data[1], 1.0f, 0.01f);
  EXPECT_NEAR(out_data[2], 2.0f, 0.01f);
  EXPECT_NEAR(out_data[3], 3.0f, 0.01f);
  // Second token (id=1) should have values [4, 5, 6, 7]
  EXPECT_NEAR(out_data[4], 4.0f, 0.01f);
  EXPECT_NEAR(out_data[5], 5.0f, 0.01f);
  EXPECT_NEAR(out_data[6], 6.0f, 0.01f);
  EXPECT_NEAR(out_data[7], 7.0f, 0.01f);
}

TEST(Float16Test, EmbeddingLookup_Fp32) {
  // Test embedding_lookup_node with fp32 weight (existing functionality)
  dynamic_tensor ids(dtype::int32, {1, 2});
  ids.data_ptr<int32_t>()[0] = 0;
  ids.data_ptr<int32_t>()[1] = 1;

  dynamic_tensor weight_fp32(dtype::float32, {10, 4});

  // Fill weight with test data
  float* w_data = weight_fp32.data_ptr<float>();
  for (int i = 0; i < 10 * 4; ++i) {
    w_data[i] = static_cast<float>(i);
  }

  embedding_lookup_node emb;
  dynamic_tensor output = emb.compute_test(ids, weight_fp32);

  EXPECT_EQ(output.get_dtype(), dtype::float32);
  ASSERT_EQ(output.shape().size(), 3);
  EXPECT_EQ(output.shape()[0], 1);
  EXPECT_EQ(output.shape()[1], 2);
  EXPECT_EQ(output.shape()[2], 4);

  // Verify embeddings
  float* out_data = output.data_ptr<float>();
  EXPECT_FLOAT_EQ(out_data[0], 0.0f);
  EXPECT_FLOAT_EQ(out_data[1], 1.0f);
  EXPECT_FLOAT_EQ(out_data[4], 4.0f);
  EXPECT_FLOAT_EQ(out_data[5], 5.0f);
}
