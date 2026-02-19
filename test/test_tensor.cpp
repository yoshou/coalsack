#include <gtest/gtest.h>

#include <numeric>

#include "coalsack/tensor/graph_proc_tensor.h"

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

TEST(TensorTest, TestCopy) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  tensor<float, 2> tensor1({3, 3}, data.data());

  tensor<float, 2> tensor2({3, 3});
  coalsack::copy<2>(tensor1.get_data(), tensor2.get_data(), tensor1.shape, tensor1.stride,
                    tensor2.stride);

  for (uint32_t i = 0; i < 3; i++) {
    for (uint32_t j = 0; j < 3; j++) {
      ASSERT_EQ(tensor1.get({i, j}), tensor2.get({i, j}));
    }
  }
}

TEST(TensorTest, TestConcat) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {5, 6, 7, 8};
  std::vector<float> data3 = {9, 10, 11, 12};

  tensor<float, 2> tensor1({2, 2}, data1.data());
  tensor<float, 2> tensor2({2, 2}, data2.data());
  tensor<float, 2> tensor3({2, 2}, data3.data());

  std::vector<tensor<float, 2>> tensors = {tensor1, tensor2, tensor3};
  auto result = tensor<float, 2>::concat<0>(tensors);

  ASSERT_EQ(result.shape[0], 6);
  ASSERT_EQ(result.shape[1], 2);
  ASSERT_EQ(result.get({0, 0}), 1);
  ASSERT_EQ(result.get({2, 0}), 5);
  ASSERT_EQ(result.get({4, 0}), 9);
}

TEST(TensorTest, TestStack) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {5, 6, 7, 8};

  tensor<float, 2> tensor1({2, 2}, data1.data());
  tensor<float, 2> tensor2({2, 2}, data2.data());
  // With stride {1, 2}: tensor1[i,j] = data1[i*1 + j*2]
  // tensor1: [0,0]=1 [0,1]=3, [1,0]=2 [1,1]=4
  // tensor2: [0,0]=5 [0,1]=7, [1,0]=6 [1,1]=8

  std::vector<tensor<float, 2>> tensors = {tensor1, tensor2};
  auto result = tensor<float, 2>::stack<1>(tensors);

  ASSERT_EQ(result.shape[0], 2);
  ASSERT_EQ(result.shape[1], 2);
  ASSERT_EQ(result.shape[2], 2);
  // stack<1> creates new dimension at position 2
  // result should concat along dim 2
  ASSERT_EQ(result.get({0, 0, 0}), 1);
  ASSERT_EQ(result.get({1, 0, 0}), 2);
  ASSERT_EQ(result.get({0, 0, 1}), 5);
  ASSERT_EQ(result.get({1, 0, 1}), 6);
}

TEST(TensorTest, TestSoftmax) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
  tensor<float, 2> tensor1({3, 2}, data.data());

  auto result = tensor1.softmax(0);

  // Check that the sum along axis 0 is approximately 1.0
  float sum0 = result.get({0, 0}) + result.get({1, 0}) + result.get({2, 0});
  float sum1 = result.get({0, 1}) + result.get({1, 1}) + result.get({2, 1});

  ASSERT_NEAR(sum0, 1.0f, 1e-5f);
  ASSERT_NEAR(sum1, 1.0f, 1e-5f);

  // Check that values are in (0, 1)
  for (uint32_t i = 0; i < 3; i++) {
    for (uint32_t j = 0; j < 2; j++) {
      float val = result.get({i, j});
      ASSERT_GT(val, 0.0f);
      ASSERT_LT(val, 1.0f);
    }
  }
}

TEST(TensorTest, TestSum) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  tensor<float, 2> tensor1({2, 3}, data.data());
  // tensor layout: [0,0]=1 [0,1]=3 [0,2]=5
  //                [1,0]=2 [1,1]=4 [1,2]=6

  auto result = tensor1.sum<1>({1});  // sum along axis 1

  ASSERT_EQ(result.shape[0], 2);
  ASSERT_EQ(result.get({0}), 9);   // 1 + 3 + 5
  ASSERT_EQ(result.get({1}), 12);  // 2 + 4 + 6
}

TEST(TensorTest, TestMax) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  tensor<float, 2> tensor1({2, 3}, data.data());
  // tensor layout: [0,0]=1 [0,1]=3 [0,2]=5
  //                [1,0]=2 [1,1]=4 [1,2]=6

  auto result = tensor1.max<1>({1});  // max along axis 1

  ASSERT_EQ(result.shape[0], 2);
  ASSERT_EQ(result.get({0}), 5);  // max(1, 3, 5)
  ASSERT_EQ(result.get({1}), 6);  // max(2, 4, 6)
}

TEST(TensorTest, TestMaxPool3D) {
  std::vector<float> data(4 * 4 * 4);
  std::iota(data.begin(), data.end(), 1.0f);

  tensor<float, 3> tensor1({4, 4, 4}, data.data());
  auto result = tensor1.max_pool3d(2, 2, 0, 1);

  ASSERT_EQ(result.shape[0], 4);
  ASSERT_EQ(result.shape[1], 4);
  ASSERT_EQ(result.shape[2], 4);

  // Check that max pooling produces larger or equal values
  ASSERT_GE(result.get({0, 0, 0}), tensor1.get({0, 0, 0}));
}

TEST(TensorTest, TestView) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  tensor<float, 3> tensor1({3, 2, 2}, data.data());
  // stride = {1, 3, 6}: tensor[i,j,k] = data[i*1 + j*3 + k*6]
  // [0,0,0]=1 [0,0,1]=7,  [0,1,0]=4 [0,1,1]=10
  // [1,0,0]=2 [1,0,1]=8,  [1,1,0]=5 [1,1,1]=11
  // [2,0,0]=3 [2,0,1]=9,  [2,1,0]=6 [2,1,1]=12

  auto view = tensor1.view<2>({2, 2}, {1, 0, 0});
  // offset moves data pointer by 1*1 + 0*3 + 0*6 = 1
  // view uses stride[0]=1, stride[1]=3 from original

  ASSERT_EQ(view.shape[0], 2);
  ASSERT_EQ(view.shape[1], 2);
  ASSERT_EQ(view.get({0, 0}), 2);  // data[1 + 0*1 + 0*3] = data[1] = 2
  ASSERT_EQ(view.get({1, 1}), 6);  // data[1 + 1*1 + 1*3] = data[5] = 6
}

TEST(TensorTest, TestSqueeze) {
  std::vector<float> data = {1, 2, 3, 4};
  tensor<float, 3> tensor1({1, 2, 2}, data.data());

  auto view = tensor1.squeeze<2>(0);

  ASSERT_EQ(view.shape[0], 2);
  ASSERT_EQ(view.shape[1], 2);
  ASSERT_EQ(view.get({0, 0}), 1);
  ASSERT_EQ(view.get({1, 1}), 4);
}

TEST(TensorTest, TestUnsqueeze) {
  std::vector<float> data = {1, 2, 3, 4};
  tensor<float, 2> tensor1({2, 2}, data.data());

  auto view = tensor1.unsqueeze<3>(1);

  ASSERT_EQ(view.shape[0], 2);
  ASSERT_EQ(view.shape[1], 1);
  ASSERT_EQ(view.shape[2], 2);
  ASSERT_EQ(view.get({0, 0, 0}), 1);
  ASSERT_EQ(view.get({1, 0, 1}), 4);
}

TEST(TensorTest, TestReshape) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  tensor<float, 2> tensor1({2, 3}, data.data());

  auto result = tensor1.view().reshape<3>({1, 2, 3});

  ASSERT_EQ(result.shape[0], 1);
  ASSERT_EQ(result.shape[1], 2);
  ASSERT_EQ(result.shape[2], 3);
  ASSERT_EQ(result.get({0, 0, 0}), 1);
  ASSERT_EQ(result.get({0, 1, 2}), 6);
}

TEST(TensorTest, TestTransformWithTwoTensors) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {5, 6, 7, 8};

  tensor<float, 2> tensor1({2, 2}, data1.data());
  tensor<float, 2> tensor2({2, 2}, data2.data());

  auto result = tensor1.transform(tensor2, [](float a, float b, auto...) { return a + b; });

  ASSERT_EQ(result.get({0, 0}), 6);   // 1 + 5
  ASSERT_EQ(result.get({1, 0}), 8);   // 2 + 6
  ASSERT_EQ(result.get({0, 1}), 10);  // 3 + 7
  ASSERT_EQ(result.get({1, 1}), 12);  // 4 + 8
}

TEST(TensorTest, TestBroadcasting) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {10};

  tensor<float, 2> tensor1({2, 2}, data1.data());
  tensor<float, 2> tensor2({1, 1}, data2.data());

  auto result = tensor1.transform(tensor2, [](float a, float b, auto...) { return a + b; });

  ASSERT_EQ(result.get({0, 0}), 11);  // 1 + 10
  ASSERT_EQ(result.get({1, 0}), 12);  // 2 + 10
  ASSERT_EQ(result.get({0, 1}), 13);  // 3 + 10
  ASSERT_EQ(result.get({1, 1}), 14);  // 4 + 10
}

TEST(TensorTest, TestZeros) {
  auto tensor1 = tensor<float, 2>::zeros({3, 3});

  for (uint32_t i = 0; i < 3; i++) {
    for (uint32_t j = 0; j < 3; j++) {
      ASSERT_EQ(tensor1.get({i, j}), 0.0f);
    }
  }
}

TEST(TensorTest, TestCast) {
  std::vector<float> data = {1.5f, 2.7f, 3.2f, 4.9f};
  tensor<float, 2> tensor1({2, 2}, data.data());

  auto tensor2 = tensor1.cast<int>();

  ASSERT_EQ(tensor2.get({0, 0}), 1);
  ASSERT_EQ(tensor2.get({1, 0}), 2);
  ASSERT_EQ(tensor2.get({0, 1}), 3);
  ASSERT_EQ(tensor2.get({1, 1}), 4);
}

TEST(TensorTest, TestEmpty) {
  tensor<float, 2> empty_tensor;
  ASSERT_TRUE(empty_tensor.empty());

  tensor<float, 2> nonempty_tensor({2, 2});
  ASSERT_FALSE(nonempty_tensor.empty());
}

TEST(TensorTest, TestGetSize) {
  tensor<float, 3> tensor1({2, 3, 4});

  ASSERT_EQ(tensor1.get_size(), 24);
  ASSERT_EQ(tensor1.get_size(0), 2);
  ASSERT_EQ(tensor1.get_size(1), 3);
  ASSERT_EQ(tensor1.get_size(2), 4);
}

TEST(TensorTest, TestCopyConstructorAndAssignment) {
  std::vector<float> data = {1, 2, 3, 4};
  tensor<float, 2> tensor1({2, 2}, data.data());

  // Copy constructor
  tensor<float, 2> tensor2(tensor1);
  ASSERT_EQ(tensor2.get({0, 0}), 1);
  ASSERT_EQ(tensor2.get({1, 1}), 4);

  // Copy assignment
  tensor<float, 2> tensor3;
  tensor3 = tensor1;
  ASSERT_EQ(tensor3.get({0, 0}), 1);
  ASSERT_EQ(tensor3.get({1, 1}), 4);
}

TEST(TensorTest, TestMoveConstructorAndAssignment) {
  std::vector<float> data = {1, 2, 3, 4};
  tensor<float, 2> tensor1({2, 2}, data.data());

  // Move constructor
  tensor<float, 2> tensor2(std::move(tensor1));
  ASSERT_EQ(tensor2.get({0, 0}), 1);
  ASSERT_EQ(tensor2.get({1, 1}), 4);

  // Move assignment
  tensor<float, 2> tensor3({2, 2}, data.data());
  tensor<float, 2> tensor4;
  tensor4 = std::move(tensor3);
  ASSERT_EQ(tensor4.get({0, 0}), 1);
  ASSERT_EQ(tensor4.get({1, 1}), 4);
}

TEST(TensorTest, TestViewAssign) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {10, 20, 30, 40};

  tensor<float, 2> tensor1({2, 2}, data1.data());
  tensor<float, 2> tensor2({2, 2}, data2.data());

  auto view1 = tensor1.view();
  auto view2 = tensor2.view();

  view1.assign(view2);

  ASSERT_EQ(tensor1.get({0, 0}), 10);
  ASSERT_EQ(tensor1.get({1, 0}), 20);
  ASSERT_EQ(tensor1.get({0, 1}), 30);
  ASSERT_EQ(tensor1.get({1, 1}), 40);
}

TEST(TensorTest, TestContiguous) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  tensor<float, 2> tensor1({2, 3}, data.data());
  // original: [0,0]=1 [0,1]=3 [0,2]=5
  //           [1,0]=2 [1,1]=4 [1,2]=6

  auto transposed = tensor1.transpose({1, 0});
  // transposed should be {3, 2}
  auto contiguous = transposed.contiguous();

  ASSERT_EQ(contiguous.shape[0], 3);
  ASSERT_EQ(contiguous.shape[1], 2);
  ASSERT_EQ(contiguous.get({0, 0}), 1);
  ASSERT_EQ(contiguous.get({0, 1}), 2);
  ASSERT_EQ(contiguous.get({1, 0}), 3);
  ASSERT_EQ(contiguous.get({1, 1}), 4);
  ASSERT_EQ(contiguous.get({2, 0}), 5);
  ASSERT_EQ(contiguous.get({2, 1}), 6);
}