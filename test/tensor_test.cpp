#include <gtest/gtest.h>

#include <numeric>

#include "graph_proc_tensor.h"

using namespace coalsack;

TEST(TensorTest, TestConstruct1) {
  std::vector<float> data = {1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0};

  tensor<float, 4> tensor({4, 3, 1, 1}, data.data());
}

TEST(TensorTest, TestConstruct2) {
  std::vector<float> data = {1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0};

  tensor<float, 4> tensor({3, 3, 1, 1}, {1, 4, 12, 1}, data.data());

  ASSERT_EQ(tensor.get({0, 0, 0, 0}), 1);
  ASSERT_EQ(tensor.get({1, 1, 0, 0}), 5);
  ASSERT_EQ(tensor.get({2, 1, 0, 0}), 6);
}

TEST(TensorTest, TestConstruct3) {
  std::vector<float> data = {1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0};

  tensor<float, 4> tensor({3, 3, 1, 1}, {1, 4, 12, 1}, data.data(), {4, 1, 12, 1});

  ASSERT_EQ(tensor.get({0, 0, 0, 0}), 1);
  ASSERT_EQ(tensor.get({1, 1, 0, 0}), 5);
  ASSERT_EQ(tensor.get({1, 2, 0, 0}), 6);
  ASSERT_EQ(tensor.get({2, 1, 0, 0}), 8);
  ASSERT_EQ(tensor.get({2, 2, 0, 0}), 9);
}

TEST(TensorTest, TestTransform1) {
  std::vector<float> data = {1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0};

  std::vector<float> addend = {1, 2, 3};

  tensor<float, 4> tensor({3, 3, 1, 1}, {1, 4, 12, 1}, data.data());

  const auto result =
      tensor.transform([addend](const float value, const size_t w, const size_t h, const size_t c,
                                const size_t n) { return value + addend[h]; });

  ASSERT_EQ(result.get({0, 0, 0, 0}), 2);
  ASSERT_EQ(result.get({1, 1, 0, 0}), 7);
  ASSERT_EQ(result.get({2, 1, 0, 0}), 8);
  ASSERT_EQ(result.get({2, 2, 0, 0}), 12);
}

TEST(TensorTest, TestTranspose1) {
  std::vector<float> data = {1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0};

  tensor<float, 4> tensor({3, 3, 1, 1}, {1, 4, 12, 1}, data.data());

  const auto result = tensor.transpose({1, 0, 2, 3}).contiguous();

  ASSERT_EQ(result.get({0, 0, 0, 0}), 1);
  ASSERT_EQ(result.get({1, 1, 0, 0}), 5);
  ASSERT_EQ(result.get({1, 2, 0, 0}), 6);
  ASSERT_EQ(result.get({2, 1, 0, 0}), 8);
  ASSERT_EQ(result.get({2, 2, 0, 0}), 9);
}

TEST(TensorTest, TestTopK1) {
  std::vector<float> data = {4, 5, 4, 1, 7, 9, 1, 2, 1, 5, 5, 4, 5, 8, 6, 3};

  constexpr auto k = 4;

  tensor<float, 1> tensor({16}, data.data());

  const auto [values, indices] = tensor.view().topk(k);

  std::array<float, k> sorted_values;
  for (uint32_t i = 0; i < k; i++) {
    sorted_values[i] = values.get({i});
  }

  std::sort(sorted_values.begin(), sorted_values.end());

  ASSERT_EQ(sorted_values[0], 6);
  ASSERT_EQ(sorted_values[1], 7);
  ASSERT_EQ(sorted_values[2], 8);
  ASSERT_EQ(sorted_values[3], 9);
}

TEST(TensorTest, TestTopK2) {
  std::vector<float> data = {4, 5, 4, 1, 7, 9, 1, 2, 1, 5, 5, 4, 5, 8, 6, 3};

  constexpr auto k = 2;

  tensor<float, 2> tensor({8, 2}, data.data());

  const auto [values, indices] = tensor.view().topk(k);

  std::array<std::array<float, k>, 2> sorted_values;
  for (uint32_t j = 0; j < 2; j++) {
    for (uint32_t i = 0; i < k; i++) {
      sorted_values[j][i] = values.get({i, j});
    }
    std::sort(sorted_values[j].begin(), sorted_values[j].end());
  }

  ASSERT_EQ(sorted_values[0][0], 7);
  ASSERT_EQ(sorted_values[0][1], 9);
  ASSERT_EQ(sorted_values[1][0], 6);
  ASSERT_EQ(sorted_values[1][1], 8);
}