#include <gtest/gtest.h>
#include <numeric>
#include "graph_proc_tensor.h"

using namespace coalsack;

TEST(TensorTest, TestConstruct1)
{
    std::vector<float> data = {
        1, 2, 3, 0,
        4, 5, 6, 0,
        7, 8, 9, 0
    };

    tensor<float, 4> tensor({4, 3, 1, 1}, data.data());
}

TEST(TensorTest, TestConstruct2)
{
    std::vector<float> data = {
        1, 2, 3, 0,
        4, 5, 6, 0,
        7, 8, 9, 0
    };

    tensor<float, 4> tensor({3, 3, 1, 1}, {1, 4, 12, 1}, data.data());

    ASSERT_EQ(tensor.get({0, 0, 0, 0}), 1);
    ASSERT_EQ(tensor.get({1, 1, 0, 0}), 5);
    ASSERT_EQ(tensor.get({2, 1, 0, 0}), 6);
}

TEST(TensorTest, TestConstruct3)
{
    std::vector<float> data = {
        1, 2, 3, 0,
        4, 5, 6, 0,
        7, 8, 9, 0
    };

    tensor<float, 4> tensor({3, 3, 1, 1}, {1, 4, 12, 1}, data.data(), {4, 1, 12, 1});
     
    ASSERT_EQ(tensor.get({0, 0, 0, 0}), 1);
    ASSERT_EQ(tensor.get({1, 1, 0, 0}), 5);
    ASSERT_EQ(tensor.get({1, 2, 0, 0}), 6);
    ASSERT_EQ(tensor.get({2, 1, 0, 0}), 8);
    ASSERT_EQ(tensor.get({2, 2, 0, 0}), 9);
}

TEST(TensorTest, TestTransform1)
{
    std::vector<float> data = {
        1, 2, 3, 0,
        4, 5, 6, 0,
        7, 8, 9, 0
    };
    
    std::vector<float> addend = {
        1, 2, 3
    };

    tensor<float, 4> tensor({3, 3, 1, 1}, {1, 4, 12, 1}, data.data());

    const auto result = tensor.transform([addend](const float value, const size_t w, const size_t h, const size_t c, const size_t n) {
        return value + addend[h];
    });

    ASSERT_EQ(result.get({0, 0, 0, 0}), 2);
    ASSERT_EQ(result.get({1, 1, 0, 0}), 7);
    ASSERT_EQ(result.get({2, 1, 0, 0}), 8);
    ASSERT_EQ(result.get({2, 2, 0, 0}), 12);
}

TEST(TensorTest, TestTranspose1)
{
    std::vector<float> data = {
        1, 2, 3, 0,
        4, 5, 6, 0,
        7, 8, 9, 0
    };

    tensor<float, 4> tensor({3, 3, 1, 1}, {1, 4, 12, 1}, data.data());

    const auto result = tensor.transpose({1, 0, 2, 3}).contiguous();
     
    ASSERT_EQ(result.get({0, 0, 0, 0}), 1);
    ASSERT_EQ(result.get({1, 1, 0, 0}), 5);
    ASSERT_EQ(result.get({1, 2, 0, 0}), 6);
    ASSERT_EQ(result.get({2, 1, 0, 0}), 8);
    ASSERT_EQ(result.get({2, 2, 0, 0}), 9);
}