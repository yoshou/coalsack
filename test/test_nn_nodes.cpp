#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "dynamic_tensor.h"
#include "dynamic_tensor_message.h"
#include "model_io_nodes.h"
#include "nn_op_node.h"
#include "result_message.h"

// nn_ops
#include "nn_ops/add_node.h"
#include "nn_ops/and_node.h"
#include "nn_ops/cast_node.h"
#include "nn_ops/clip_node.h"
#include "nn_ops/concat_node.h"
#include "nn_ops/constant_node.h"
#include "nn_ops/constant_of_shape_node.h"
#include "nn_ops/conv_node.h"
#include "nn_ops/cos_node.h"
#include "nn_ops/div_node.h"
#include "nn_ops/equal_node.h"
#include "nn_ops/erf_node.h"
#include "nn_ops/exp_node.h"
#include "nn_ops/expand_node.h"
#include "nn_ops/flatten_node.h"
#include "nn_ops/gather_elements_node.h"
#include "nn_ops/gather_node.h"
#include "nn_ops/greater_node.h"
#include "nn_ops/grid_sample_node.h"
#include "nn_ops/layer_normalization_node.h"
#include "nn_ops/log_node.h"
#include "nn_ops/log_softmax_node.h"
#include "nn_ops/matmul_node.h"
#include "nn_ops/matmul_transpose_mixed_node.h"
#include "nn_ops/max_pool_node.h"
#include "nn_ops/mod_node.h"
#include "nn_ops/mul_node.h"
#include "nn_ops/neg_node.h"
#include "nn_ops/non_zero_node.h"
#include "nn_ops/not_node.h"
#include "nn_ops/or_node.h"
#include "nn_ops/pow_node.h"
#include "nn_ops/range_node.h"
#include "nn_ops/reduce_l2_node.h"
#include "nn_ops/reduce_mean_node.h"
#include "nn_ops/relu_node.h"
#include "nn_ops/reshape_node.h"
#include "nn_ops/scatter_nd_node.h"
#include "nn_ops/shape_node.h"
#include "nn_ops/sigmoid_node.h"
#include "nn_ops/sin_node.h"
#include "nn_ops/slice_node.h"
#include "nn_ops/softmax_node.h"
#include "nn_ops/split_node.h"
#include "nn_ops/sqrt_node.h"
#include "nn_ops/squeeze_node.h"
#include "nn_ops/sub_node.h"
#include "nn_ops/tanh_node.h"
#include "nn_ops/tile_node.h"
#include "nn_ops/topk_node.h"
#include "nn_ops/transpose_node.h"
#include "nn_ops/unsqueeze_node.h"
#include "nn_ops/where_node.h"

using namespace coalsack;

static std::shared_ptr<result_message> make_result_message(
    const std::unordered_map<std::string, dynamic_tensor>& tensors, uint64_t frame_number = 1) {
  std::unordered_map<std::string, graph_message_ptr> fields;
  for (const auto& [name, tensor] : tensors) {
    auto msg = std::make_shared<dynamic_tensor_message>(tensor);
    fields[name] = msg;
  }
  auto result = result_message::ok(fields);
  result->set_frame_number(frame_number);
  return result;
}

static void expect_ok_result_message(const std::shared_ptr<graph_message>& msg) {
  auto result = std::dynamic_pointer_cast<result_message>(msg);
  ASSERT_TRUE(result) << "output is not result_message";
  ASSERT_TRUE(result->is_ok()) << "output is error result_message: " << result->get_error_message();
}

static void expect_frame_number(const std::shared_ptr<graph_message>& msg,
                                uint64_t expected_frame) {
  auto result = std::dynamic_pointer_cast<result_message>(msg);
  ASSERT_TRUE(result) << "output is not result_message";
  EXPECT_EQ(result->get_frame_number(), expected_frame);
}

static void expect_output_field(const std::shared_ptr<graph_message>& msg,
                                const std::string& expected_field_name) {
  auto result = std::dynamic_pointer_cast<result_message>(msg);
  ASSERT_TRUE(result) << "output is not result_message";

  auto field = result->get_field(expected_field_name);
  ASSERT_TRUE(field) << "missing field: " << expected_field_name;

  auto tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(field);
  ASSERT_TRUE(tensor_msg) << "field '" << expected_field_name << "' is not dynamic_tensor_message";
}

static void run_unary_test(const std::shared_ptr<graph_node>& node, const dynamic_tensor& input,
                           const std::string& expected_output_field) {
  const uint64_t input_frame = 7;

  // Set input and output names for unary_op_node
  if (auto unary_node = std::dynamic_pointer_cast<unary_op_node>(node)) {
    unary_node->set_input_name("input");
    unary_node->set_output_name(expected_output_field);
  }

  std::shared_ptr<graph_message> output_msg;
  node->get_output()->set_callback(
      std::make_shared<graph_message_callback>([&](graph_message_ptr msg) { output_msg = msg; }));

  auto input_msg = make_result_message({{"input", input}}, input_frame);
  node->process("default", input_msg);

  ASSERT_TRUE(output_msg) << "no output";
  expect_ok_result_message(output_msg);
  expect_frame_number(output_msg, input_frame);
  expect_output_field(output_msg, expected_output_field);
}

static void run_binary_test(const std::shared_ptr<graph_node>& node, const dynamic_tensor& a,
                            const dynamic_tensor& b, const std::string& input_a_name,
                            const std::string& input_b_name,
                            const std::string& expected_output_field) {
  const uint64_t input_frame = 7;

  // Set input names and output name for binary_op_node
  if (auto binary_node = std::dynamic_pointer_cast<binary_op_node>(node)) {
    binary_node->set_input_names(input_a_name, input_b_name);
    binary_node->set_output_name(expected_output_field);
  }

  std::shared_ptr<graph_message> output_msg;
  node->get_output()->set_callback(
      std::make_shared<graph_message_callback>([&](graph_message_ptr msg) { output_msg = msg; }));

  auto input_msg = make_result_message({{input_a_name, a}, {input_b_name, b}}, input_frame);
  node->process("default", input_msg);

  ASSERT_TRUE(output_msg) << "no output";
  expect_ok_result_message(output_msg);
  expect_frame_number(output_msg, input_frame);
  expect_output_field(output_msg, expected_output_field);
}

static void run_variadic_test(const std::shared_ptr<graph_node>& node,
                              const std::vector<dynamic_tensor>& inputs,
                              const std::string& expected_output_field) {
  const uint64_t input_frame = 7;

  std::unordered_map<std::string, dynamic_tensor> fields;
  std::vector<std::string> input_names;
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::string name = "input" + std::to_string(i);
    fields[name] = inputs[i];
    input_names.push_back(name);
  }

  // Set input and output names for variadic_op_node
  if (auto variadic_node = std::dynamic_pointer_cast<variadic_op_node>(node)) {
    variadic_node->set_input_names(input_names);
    variadic_node->set_output_name(expected_output_field);
  }

  std::shared_ptr<graph_message> output_msg;
  node->get_output()->set_callback(
      std::make_shared<graph_message_callback>([&](graph_message_ptr msg) { output_msg = msg; }));

  auto input_msg = make_result_message(fields, input_frame);
  node->process("default", input_msg);

  ASSERT_TRUE(output_msg) << "no output";
  expect_ok_result_message(output_msg);
  expect_frame_number(output_msg, input_frame);
  expect_output_field(output_msg, expected_output_field);
}

static void run_graphnode_result_fields_test(
    const std::shared_ptr<graph_node>& node,
    const std::unordered_map<std::string, dynamic_tensor>& fields,
    const std::string& expected_output_field) {
  const uint64_t input_frame = 7;

  std::shared_ptr<graph_message> output_msg;
  node->get_output()->set_callback(
      std::make_shared<graph_message_callback>([&](graph_message_ptr msg) { output_msg = msg; }));

  auto input_msg = make_result_message(fields, input_frame);
  node->process("default", input_msg);

  ASSERT_TRUE(output_msg) << "no output";
  expect_ok_result_message(output_msg);
  expect_frame_number(output_msg, input_frame);
  expect_output_field(output_msg, expected_output_field);
}

// Unary nodes test fixture
class UnaryNodeTest : public ::testing::Test {
 protected:
  dynamic_tensor input_f;
  dynamic_tensor input_b;
  dynamic_tensor input_i;
  dynamic_tensor input_4d;

  void SetUp() override {
    input_f = dynamic_tensor(dtype::float32, {3}, std::vector<float>{1.0f, -2.0f, 3.0f}.data());
    const bool input_b_data[] = {true, false, true};
    input_b = dynamic_tensor(dtype::bool_, {3}, input_b_data);
    input_i = dynamic_tensor(dtype::int64, {2}, std::vector<int64_t>{2, 2}.data());
    input_4d = dynamic_tensor(dtype::float32, {1, 1, 2, 2}, std::vector<float>{1, 2, 3, 4}.data());
  }
};

TEST_F(UnaryNodeTest, ReluNode) {
  run_unary_test(std::make_shared<relu_node>(), input_f, "/test/Relu_output_0");
}

TEST_F(UnaryNodeTest, TanhNode) {
  run_unary_test(std::make_shared<tanh_node>(), input_f, "/test/Tanh_output_0");
}

TEST_F(UnaryNodeTest, NegNode) {
  run_unary_test(std::make_shared<neg_node>(), input_f, "/test/Neg_output_0");
}

TEST_F(UnaryNodeTest, SqrtNode) {
  run_unary_test(std::make_shared<sqrt_node>(), input_f, "/test/Sqrt_output_0");
}

TEST_F(UnaryNodeTest, ExpNode) {
  run_unary_test(std::make_shared<exp_node>(), input_f, "/test/Exp_output_0");
}

TEST_F(UnaryNodeTest, LogNode) {
  run_unary_test(std::make_shared<log_node>(), input_f, "/test/Log_output_0");
}

TEST_F(UnaryNodeTest, SinNode) {
  run_unary_test(std::make_shared<sin_node>(), input_f, "/test/Sin_output_0");
}

TEST_F(UnaryNodeTest, CosNode) {
  run_unary_test(std::make_shared<cos_node>(), input_f, "/test/Cos_output_0");
}

TEST_F(UnaryNodeTest, ErfNode) {
  run_unary_test(std::make_shared<erf_node>(), input_f, "/test/Erf_output_0");
}

TEST_F(UnaryNodeTest, SigmoidNode) {
  run_unary_test(std::make_shared<sigmoid_node>(), input_f, "/test/Sigmoid_output_0");
}

TEST_F(UnaryNodeTest, SoftmaxNode) {
  run_unary_test(std::make_shared<softmax_node>(), input_f, "/test/Softmax_output_0");
}

TEST_F(UnaryNodeTest, LogSoftmaxNode) {
  run_unary_test(std::make_shared<log_softmax_node>(), input_f, "/test/LogSoftmax_output_0");
}

TEST_F(UnaryNodeTest, ReduceMeanNode) {
  run_unary_test(std::make_shared<reduce_mean_node>(), input_f, "/test/ReduceMean_output_0");
}

TEST_F(UnaryNodeTest, ReduceL2Node) {
  run_unary_test(std::make_shared<reduce_l2_node>(), input_f, "/test/ReduceL2_output_0");
}

TEST_F(UnaryNodeTest, NonZeroNode) {
  run_unary_test(std::make_shared<non_zero_node>(), input_f, "/test/NonZero_output_0");
}

TEST_F(UnaryNodeTest, ShapeNode) {
  run_unary_test(std::make_shared<shape_node>(), input_f, "/test/Shape_output_0");
}

TEST_F(UnaryNodeTest, FlattenNode) {
  run_unary_test(std::make_shared<flatten_node>(), input_f, "/test/Flatten_output_0");
}

TEST_F(UnaryNodeTest, TransposeNode) {
  run_unary_test(std::make_shared<transpose_node>(), input_f, "/test/Transpose_output_0");
}

TEST_F(UnaryNodeTest, ConstantOfShapeNode) {
  run_unary_test(std::make_shared<constant_of_shape_node>(), input_i,
                 "/test/ConstantOfShape_output_0");
}

TEST_F(UnaryNodeTest, NotNode) {
  run_unary_test(std::make_shared<not_node>(), input_b, "/test/Not_output_0");
}

TEST_F(UnaryNodeTest, MaxPoolNode) {
  run_unary_test(std::make_shared<max_pool_node>(), input_4d, "/test/MaxPool_output_0");
}

TEST_F(UnaryNodeTest, CastNode) {
  auto cast = std::make_shared<cast_node>();
  cast->set_target_dtype(dtype::int64);
  run_unary_test(cast, input_f, "/test/Cast_output_0");
}

TEST_F(UnaryNodeTest, SplitNode) {
  auto split = std::make_shared<split_node>();
  split->set_input_name("input");
  split->set_axis(0);
  split->set_splits({1, 2});
  split->set_output_names({"/test/Split_output_0", "/test/Split_output_1"});

  std::shared_ptr<graph_message> split_out;
  split->get_output("default")->set_callback(
      std::make_shared<graph_message_callback>([&](graph_message_ptr msg) { split_out = msg; }));

  auto split_input = dynamic_tensor(dtype::float32, {3}, std::vector<float>{1, 2, 3}.data());
  const uint64_t split_frame = 7;
  split->process("default", make_result_message({{"input", split_input}}, split_frame));

  ASSERT_TRUE(split_out);
  expect_ok_result_message(split_out);
  expect_frame_number(split_out, split_frame);
  expect_output_field(split_out, "/test/Split_output_0");
  expect_output_field(split_out, "/test/Split_output_1");
}

// Binary nodes test fixture
class BinaryNodeTest : public ::testing::Test {
 protected:
  dynamic_tensor a_f, b_f;
  dynamic_tensor a_b, b_b;
  dynamic_tensor a_i, b_i;

  void SetUp() override {
    a_f = dynamic_tensor(dtype::float32, {2}, std::vector<float>{2.0f, 3.0f}.data());
    b_f = dynamic_tensor(dtype::float32, {2}, std::vector<float>{4.0f, 5.0f}.data());
    const bool a_b_data[] = {true, false};
    const bool b_b_data[] = {false, true};
    a_b = dynamic_tensor(dtype::bool_, {2}, a_b_data);
    b_b = dynamic_tensor(dtype::bool_, {2}, b_b_data);
    a_i = dynamic_tensor(dtype::int64, {2}, std::vector<int64_t>{5, 10}.data());
    b_i = dynamic_tensor(dtype::int64, {2}, std::vector<int64_t>{2, 3}.data());
  }
};

TEST_F(BinaryNodeTest, AddNode) {
  run_binary_test(std::make_shared<add_node>(), a_f, b_f, "/test/add_input_a", "/test/add_input_b",
                  "/test/Add_output_0");
}

TEST_F(BinaryNodeTest, SubNode) {
  run_binary_test(std::make_shared<sub_node>(), a_f, b_f, "/test/sub_input_a", "/test/sub_input_b",
                  "/test/Sub_output_0");
}

TEST_F(BinaryNodeTest, MulNode) {
  run_binary_test(std::make_shared<mul_node>(), a_f, b_f, "/test/mul_input_a", "/test/mul_input_b",
                  "/test/Mul_output_0");
}

TEST_F(BinaryNodeTest, DivNode) {
  run_binary_test(std::make_shared<div_node>(), a_f, b_f, "/test/div_input_a", "/test/div_input_b",
                  "/test/Div_output_0");
}

TEST_F(BinaryNodeTest, PowNode) {
  run_binary_test(std::make_shared<pow_node>(), a_f, b_f, "/test/pow_input_a", "/test/pow_input_b",
                  "/test/Pow_output_0");
}

TEST_F(BinaryNodeTest, ModNode) {
  run_binary_test(std::make_shared<mod_node>(), a_i, b_i, "/test/mod_input_a", "/test/mod_input_b",
                  "/test/Mod_output_0");
}

TEST_F(BinaryNodeTest, AndNode) {
  run_binary_test(std::make_shared<and_node>(), a_b, b_b, "/test/and_input_a", "/test/and_input_b",
                  "/test/And_output_0");
}

TEST_F(BinaryNodeTest, OrNode) {
  run_binary_test(std::make_shared<or_node>(), a_b, b_b, "/test/or_input_a", "/test/or_input_b",
                  "/test/Or_output_0");
}

TEST_F(BinaryNodeTest, EqualNode) {
  run_binary_test(std::make_shared<equal_node>(), a_f, b_f, "/test/equal_input_a",
                  "/test/equal_input_b", "/test/Equal_output_0");
}

TEST_F(BinaryNodeTest, GreaterNode) {
  run_binary_test(std::make_shared<greater_node>(), a_f, b_f, "/test/greater_input_a",
                  "/test/greater_input_b", "/test/Greater_output_0");
}

TEST_F(BinaryNodeTest, MatmulNode) {
  run_binary_test(std::make_shared<matmul_node>(),
                  dynamic_tensor(dtype::float32, {2, 2}, std::vector<float>{1, 2, 3, 4}.data()),
                  dynamic_tensor(dtype::float32, {2, 2}, std::vector<float>{5, 6, 7, 8}.data()),
                  "/test/matmul_input_a", "/test/matmul_input_b", "/test/MatMul_output_0");
}

TEST_F(BinaryNodeTest, TileNode) {
  run_binary_test(std::make_shared<tile_node>(),
                  dynamic_tensor(dtype::float32, {2}, std::vector<float>{1, 2}.data()),
                  dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{2}.data()),
                  "/test/tile_input", "/test/tile_repeats", "/test/Tile_output_0");
}

TEST_F(BinaryNodeTest, ExpandNode) {
  run_binary_test(std::make_shared<expand_node>(),
                  dynamic_tensor(dtype::float32, {1}, std::vector<float>{1}.data()),
                  dynamic_tensor(dtype::int64, {2}, std::vector<int64_t>{2, 2}.data()),
                  "/test/expand_input", "/test/expand_shape", "/test/Expand_output_0");
}

TEST_F(BinaryNodeTest, GatherElementsNode) {
  run_binary_test(std::make_shared<gather_elements_node>(),
                  dynamic_tensor(dtype::float32, {2}, std::vector<float>{10, 20}.data()),
                  dynamic_tensor(dtype::int64, {2}, std::vector<int64_t>{1, 0}.data()),
                  "/test/gather_elements_data", "/test/gather_elements_indices",
                  "/test/GatherElements_output_0");
}

TEST_F(BinaryNodeTest, GridSampleNode) {
  run_binary_test(
      std::make_shared<grid_sample_node>(),
      dynamic_tensor(dtype::float32, {1, 1, 2, 2}, std::vector<float>{1, 2, 3, 4}.data()),
      dynamic_tensor(dtype::float32, {1, 1, 1, 2}, std::vector<float>{0.0f, 0.0f}.data()),
      "/test/grid_sample_input", "/test/grid_sample_grid", "/test/GridSample_output_0");
}

TEST_F(BinaryNodeTest, TopkNode) {
  auto topk = std::make_shared<topk_node>();
  topk->set_input_names("/test/topk_input", "/test/topk_k");
  topk->set_values_output_name("/test/TopK_output_0");
  topk->set_indices_output_name("/test/TopK_output_1");

  std::shared_ptr<graph_message> topk_out;
  topk->get_output("default")->set_callback(
      std::make_shared<graph_message_callback>([&](graph_message_ptr msg) { topk_out = msg; }));

  const uint64_t topk_frame = 7;
  auto topk_input = make_result_message(
      {{"/test/topk_input",
        dynamic_tensor(dtype::float32, {3}, std::vector<float>{3, 1, 2}.data())},
       {"/test/topk_k", dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{2}.data())}},
      topk_frame);
  topk->process("default", topk_input);

  ASSERT_TRUE(topk_out);
  expect_ok_result_message(topk_out);
  expect_frame_number(topk_out, topk_frame);
  expect_output_field(topk_out, "/test/TopK_output_0");
  expect_output_field(topk_out, "/test/TopK_output_1");
}

// Variadic nodes tests
TEST(VariadicNodeTest, ConvNode) {
  auto conv = std::make_shared<conv_node>();
  run_variadic_test(
      conv,
      {dynamic_tensor(dtype::float32, {1, 1, 3, 3},
                      std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}.data()),
       dynamic_tensor(dtype::float32, {1, 1, 2, 2}, std::vector<float>{1, 0, 0, 1}.data()),
       dynamic_tensor(dtype::float32, {1}, std::vector<float>{0}.data())},
      "/test/Conv_output_0");
}

TEST(VariadicNodeTest, ConcatNode) {
  auto concat = std::make_shared<concat_node>(2);
  run_variadic_test(concat,
                    {dynamic_tensor(dtype::float32, {1}, std::vector<float>{1}.data()),
                     dynamic_tensor(dtype::float32, {1}, std::vector<float>{2}.data())},
                    "/test/Concat_output_0");
}

TEST(VariadicNodeTest, ScatterNdNode) {
  auto scatter = std::make_shared<scatter_nd_node>();
  run_variadic_test(scatter,
                    {dynamic_tensor(dtype::float32, {3}, std::vector<float>{0, 0, 0}.data()),
                     dynamic_tensor(dtype::int64, {1, 1}, std::vector<int64_t>{1}.data()),
                     dynamic_tensor(dtype::float32, {1}, std::vector<float>{9}.data())},
                    "/test/ScatterND_output_0");
}

TEST(VariadicNodeTest, ClipNode) {
  auto clip = std::make_shared<clip_node>();
  run_variadic_test(clip,
                    {dynamic_tensor(dtype::float32, {3}, std::vector<float>{-1, 0.5f, 2}.data()),
                     dynamic_tensor(dtype::float32, {1}, std::vector<float>{0}.data()),
                     dynamic_tensor(dtype::float32, {1}, std::vector<float>{1}.data())},
                    "/test/Clip_output_0");
}

TEST(VariadicNodeTest, RangeNode) {
  auto range = std::make_shared<range_node>();
  run_variadic_test(range,
                    {dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{0}.data()),
                     dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{3}.data()),
                     dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{1}.data())},
                    "/test/Range_output_0");
}

TEST(VariadicNodeTest, WhereNode) {
  auto where = std::make_shared<where_node>();
  const bool where_cond_data[] = {true, false};
  run_variadic_test(where,
                    {dynamic_tensor(dtype::bool_, {2}, where_cond_data),
                     dynamic_tensor(dtype::float32, {2}, std::vector<float>{1, 2}.data()),
                     dynamic_tensor(dtype::float32, {2}, std::vector<float>{3, 4}.data())},
                    "/test/Where_output_0");
}

TEST(VariadicNodeTest, LayerNormalizationNode) {
  auto layer_norm = std::make_shared<layer_normalization_node>();
  run_variadic_test(layer_norm,
                    {dynamic_tensor(dtype::float32, {1, 2}, std::vector<float>{1, 2}.data()),
                     dynamic_tensor(dtype::float32, {2}, std::vector<float>{1, 1}.data()),
                     dynamic_tensor(dtype::float32, {2}, std::vector<float>{0, 0}.data())},
                    "/test/LayerNormalization_output_0");
}

// Graph-node ops tests
TEST(GraphNodeOpsTest, GatherNode) {
  auto gather = std::make_shared<gather_node>();
  gather->set_input_names("data", "indices");
  gather->set_output_name("/test/Gather_output_0");
  run_graphnode_result_fields_test(
      gather,
      {{"data", dynamic_tensor(dtype::float32, {3}, std::vector<float>{1, 2, 3}.data())},
       {"indices", dynamic_tensor(dtype::int64, {2}, std::vector<int64_t>{0, 2}.data())}},
      "/test/Gather_output_0");
}

TEST(GraphNodeOpsTest, ReshapeNode) {
  auto reshape = std::make_shared<reshape_node>();
  reshape->set_input_names("data", "shape");
  reshape->set_output_name("/test/Reshape_output_0");
  run_graphnode_result_fields_test(
      reshape,
      {{"data", dynamic_tensor(dtype::float32, {2, 2}, std::vector<float>{1, 2, 3, 4}.data())},
       {"shape", dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{4}.data())}},
      "/test/Reshape_output_0");
}

TEST(GraphNodeOpsTest, SliceNode) {
  auto slice = std::make_shared<slice_node>();
  slice->set_input_names("data", "starts", "ends");
  slice->set_output_name("/test/Slice_output_0");
  run_graphnode_result_fields_test(
      slice,
      {{"data", dynamic_tensor(dtype::float32, {4}, std::vector<float>{1, 2, 3, 4}.data())},
       {"starts", dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{1}.data())},
       {"ends", dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{3}.data())}},
      "/test/Slice_output_0");
}

TEST(GraphNodeOpsTest, SqueezeNode) {
  auto squeeze = std::make_shared<squeeze_node>();
  squeeze->set_input_names("data", "axes");
  squeeze->set_output_name("/test/Squeeze_output_0");
  run_graphnode_result_fields_test(
      squeeze,
      {{"data", dynamic_tensor(dtype::float32, {1, 2}, std::vector<float>{1, 2}.data())},
       {"axes", dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{0}.data())}},
      "/test/Squeeze_output_0");
}

TEST(GraphNodeOpsTest, UnsqueezeNode) {
  auto unsqueeze = std::make_shared<unsqueeze_node>();
  unsqueeze->set_input_names("data", "axes");
  unsqueeze->set_output_name("/test/Unsqueeze_output_0");
  run_graphnode_result_fields_test(
      unsqueeze,
      {{"data", dynamic_tensor(dtype::float32, {2}, std::vector<float>{1, 2}.data())},
       {"axes", dynamic_tensor(dtype::int64, {1}, std::vector<int64_t>{0}.data())}},
      "/test/Unsqueeze_output_0");
}

TEST(GraphNodeOpsTest, ConstantNode) {
  auto constant = std::make_shared<constant_node>(
      dynamic_tensor(dtype::float32, {1}, std::vector<float>{42}.data()), "test_constant_output");

  const uint64_t const_frame = 7;
  std::shared_ptr<graph_message> const_output_msg;
  constant->get_output()->set_callback(std::make_shared<graph_message_callback>(
      [&](graph_message_ptr msg) { const_output_msg = msg; }));

  auto const_input = make_result_message(
      {{"dummy", dynamic_tensor(dtype::float32, {1}, std::vector<float>{0}.data())}}, const_frame);
  constant->process("default", const_input);

  ASSERT_TRUE(const_output_msg);
  expect_ok_result_message(const_output_msg);
  expect_frame_number(const_output_msg, const_frame);
  expect_output_field(const_output_msg, "test_constant_output");
}

// Model IO nodes tests
TEST(ModelIoNodesTest, ModelInputNode) {
  auto model_input = std::make_shared<model_input_node>();
  model_input->set_tensor("input",
                          dynamic_tensor(dtype::float32, {1}, std::vector<float>{1}.data()));
  model_input->set_frame_number(7);

  std::shared_ptr<graph_message> model_input_out;
  model_input->get_output()->set_callback(std::make_shared<graph_message_callback>(
      [&](graph_message_ptr msg) { model_input_out = msg; }));
  model_input->run();

  ASSERT_TRUE(model_input_out);
  expect_ok_result_message(model_input_out);
  expect_output_field(model_input_out, "input");
}

TEST(ModelIoNodesTest, ModelOutputNode) {
  auto model_output = std::make_shared<model_output_node>();
  bool output_called = false;
  model_output->set_callback([&](const std::unordered_map<std::string, dynamic_tensor>& outputs) {
    output_called = !outputs.empty();
  });

  auto out_msg = make_result_message(
      {{"out", dynamic_tensor(dtype::float32, {1}, std::vector<float>{2}.data())}}, 3);

  model_output->process("default", out_msg);

  EXPECT_TRUE(output_called) << "callback not called";
}

// =============================================================================
// matmul_transpose_mixed_node tests
// =============================================================================

// Naive reference: C[i][j] = sum_k A[i][k] * B[j][k]
static void matmul_transpose_ref(const float* a, const float* b, float* out, int64_t M, int64_t N,
                                 int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        sum += static_cast<double>(a[i * K + k]) * static_cast<double>(b[j * K + k]);
      }
      out[i * N + j] = static_cast<float>(sum);
    }
  }
}

TEST(MatmulTransposeMixedTest, SmallKnownValues) {
  // A: [2, 3], B: [4, 3] -> Output: [2, 4] = A @ B.T
  auto a = dynamic_tensor(dtype::float32, {2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}.data());
  auto b = dynamic_tensor(dtype::float32, {4, 3},
                          std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}.data());

  matmul_transpose_mixed_node node;
  auto result = node.compute_test(a, b);

  ASSERT_EQ(result.shape(), (std::vector<int64_t>{2, 4}));
  const float* out = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(out[0], 1.0f);
  EXPECT_FLOAT_EQ(out[1], 2.0f);
  EXPECT_FLOAT_EQ(out[2], 3.0f);
  EXPECT_FLOAT_EQ(out[3], 6.0f);
  EXPECT_FLOAT_EQ(out[4], 4.0f);
  EXPECT_FLOAT_EQ(out[5], 5.0f);
  EXPECT_FLOAT_EQ(out[6], 6.0f);
  EXPECT_FLOAT_EQ(out[7], 15.0f);
}

TEST(MatmulTransposeMixedTest, RandomLarge) {
  const int64_t M = 4, K = 256, N = 256;
  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  dynamic_tensor a(dtype::float32, {M, K});
  dynamic_tensor b(dtype::float32, {N, K});
  for (int64_t i = 0; i < M * K; ++i) a.data_ptr<float>()[i] = dis(gen);
  for (int64_t i = 0; i < N * K; ++i) b.data_ptr<float>()[i] = dis(gen);

  std::vector<float> ref(M * N);
  matmul_transpose_ref(a.data_ptr<float>(), b.data_ptr<float>(), ref.data(), M, N, K);

  matmul_transpose_mixed_node node;
  auto result = node.compute_test(a, b);

  ASSERT_EQ(result.shape(), (std::vector<int64_t>{M, N}));
  const float* out = result.data_ptr<float>();
  for (int64_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(out[i], ref[i], 1e-4f * std::max(1.0f, std::abs(ref[i])))
        << "mismatch at index " << i;
  }
}

TEST(MatmulTransposeMixedTest, NonAlignedDimensions) {
  // K=13 (not divisible by 8/16), N=5 (not divisible by 4)
  const int64_t M = 3, K = 13, N = 5;
  std::mt19937 gen(99);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  dynamic_tensor a(dtype::float32, {M, K});
  dynamic_tensor b(dtype::float32, {N, K});
  for (int64_t i = 0; i < M * K; ++i) a.data_ptr<float>()[i] = dis(gen);
  for (int64_t i = 0; i < N * K; ++i) b.data_ptr<float>()[i] = dis(gen);

  std::vector<float> ref(M * N);
  matmul_transpose_ref(a.data_ptr<float>(), b.data_ptr<float>(), ref.data(), M, N, K);

  matmul_transpose_mixed_node node;
  auto result = node.compute_test(a, b);

  ASSERT_EQ(result.shape(), (std::vector<int64_t>{M, N}));
  const float* out = result.data_ptr<float>();
  for (int64_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(out[i], ref[i], 1e-4f * std::max(1.0f, std::abs(ref[i])))
        << "mismatch at index " << i;
  }
}

TEST(MatmulTransposeMixedTest, SingleRow) {
  // M=1: most common case in LLM autoregressive inference
  const int64_t M = 1, K = 768, N = 768;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

  dynamic_tensor a(dtype::float32, {M, K});
  dynamic_tensor b(dtype::float32, {N, K});
  for (int64_t i = 0; i < M * K; ++i) a.data_ptr<float>()[i] = dis(gen);
  for (int64_t i = 0; i < N * K; ++i) b.data_ptr<float>()[i] = dis(gen);

  std::vector<float> ref(M * N);
  matmul_transpose_ref(a.data_ptr<float>(), b.data_ptr<float>(), ref.data(), M, N, K);

  matmul_transpose_mixed_node node;
  auto result = node.compute_test(a, b);

  ASSERT_EQ(result.shape(), (std::vector<int64_t>{M, N}));
  const float* out = result.data_ptr<float>();
  for (int64_t i = 0; i < M * N; ++i) {
    EXPECT_NEAR(out[i], ref[i], 1e-4f * std::max(1.0f, std::abs(ref[i])))
        << "mismatch at index " << i;
  }
}
