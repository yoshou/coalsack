#include <gtest/gtest.h>

#include <map>
#include <memory>

#include "coalsack/core/graph_edge.h"
#include "coalsack/core/graph_message.h"
#include "coalsack/core/graph_node.h"
#include "coalsack/core/graph_proc.h"
#include "coalsack/image/frame_message.h"
#include "coalsack/image/image_nodes.h"
#include "coalsack/nn/nn_op_node.h"
#include "coalsack/nn/result_message.h"
#include "coalsack/nn/result_message_nodes.h"
#include "coalsack/tensor/dynamic_tensor.h"
#include "coalsack/tensor/dynamic_tensor_message.h"
#include "coalsack/util/syncer.h"

using namespace coalsack;

class test_input_node : public graph_node {
 private:
  graph_edge_ptr output_;
  std::vector<std::tuple<std::string, uint64_t, std::shared_ptr<dynamic_tensor_message>>>
      frame_data_;

 public:
  test_input_node() : graph_node(), output_(std::make_shared<graph_edge>(this)) {
    set_output(output_);
  }

  std::string get_proc_name() const override { return "test_input_node"; }

  void add_frame(uint64_t frame_number,
                 const std::unordered_map<std::string, dynamic_tensor>& fields) {
    for (const auto& [field_name, tensor] : fields) {
      auto msg = std::make_shared<dynamic_tensor_message>(tensor);
      frame_data_.push_back({field_name, frame_number, msg});
    }
  }

  void run() override {
    std::map<uint64_t, std::unordered_map<std::string, graph_message_ptr>> frames_by_number;

    for (const auto& [field_name, frame_number, msg] : frame_data_) {
      frames_by_number[frame_number][field_name] = msg;
    }

    for (const auto& [frame_number, fields] : frames_by_number) {
      auto result = result_message::ok(fields);
      result->set_frame_number(frame_number);
      result->set_timestamp(static_cast<double>(frame_number));
      output_->send(result);
    }
  }

  void process(std::string input_name, graph_message_ptr message) override {}
};

class test_add_node : public binary_op_node {
 protected:
  dynamic_tensor compute(const dynamic_tensor& a, const dynamic_tensor& b) override {
    if (a.shape() != b.shape()) {
      throw std::runtime_error("Shape mismatch");
    }

    dynamic_tensor result(a.get_dtype(), a.shape());

    if (a.get_dtype() == dtype::float32) {
      const float* a_data = a.data_ptr<float>();
      const float* b_data = b.data_ptr<float>();
      float* result_data = result.data_ptr<float>();

      for (size_t i = 0; i < a.numel(); ++i) {
        result_data[i] = a_data[i] + b_data[i];
      }
    }

    return result;
  }

 public:
  test_add_node() : binary_op_node("add") {
    set_input_a_name("input_a");
    set_input_b_name("input_b");
    set_output_name("output");
  }

  void set_input_a_name(const std::string& name) { input_a_name_ = name; }

  void set_input_b_name(const std::string& name) { input_b_name_ = name; }
};

TEST(FrameSyncTest, BasicResultMessageSyncNode) {
  graph_proc proc;

  auto input_node = std::make_shared<test_input_node>();
  proc.get_graph()->add_node(input_node);

  auto extractor = std::make_shared<result_field_extractor_node>();
  extractor->set_input(input_node->get_output());
  proc.get_graph()->add_node(extractor);

  auto edge_a = extractor->add_output("input_a");
  auto edge_b = extractor->add_output("input_b");

  auto sync_node = std::make_shared<result_message_sync_node>();
  sync_node->set_input(edge_a, "input_a");
  sync_node->set_input(edge_b, "input_b");
  sync_node->set_initial_ids({"input_a", "input_b"});
  proc.get_graph()->add_node(sync_node);

  bool received = false;
  std::shared_ptr<result_message> result_msg;

  auto output_edge = sync_node->get_output();
  output_edge->set_callback(std::make_shared<graph_message_callback>([&](graph_message_ptr msg) {
    result_msg = std::dynamic_pointer_cast<result_message>(msg);
    received = true;
  }));

  std::vector<float> data_a(4, 1.0f);
  std::vector<float> data_b(4, 2.0f);
  input_node->add_frame(1, {{"input_a", dynamic_tensor(dtype::float32, {2, 2}, data_a.data())},
                            {"input_b", dynamic_tensor(dtype::float32, {2, 2}, data_b.data())}});

  proc.run();

  ASSERT_TRUE(received);
  ASSERT_TRUE(result_msg);
  ASSERT_TRUE(result_msg->is_ok());

  auto field_a = result_msg->get_field("input_a");
  auto field_b = result_msg->get_field("input_b");

  ASSERT_TRUE(field_a);
  ASSERT_TRUE(field_b);
}

TEST(FrameSyncTest, BinaryOpWithSync) {
  graph_proc proc;

  auto input_node = std::make_shared<test_input_node>();
  proc.get_graph()->add_node(input_node);

  auto extractor = std::make_shared<result_field_extractor_node>();
  extractor->set_input(input_node->get_output());
  proc.get_graph()->add_node(extractor);

  auto edge_a = extractor->add_output("input_a");
  auto edge_b = extractor->add_output("input_b");

  auto sync_node = std::make_shared<result_message_sync_node>();
  sync_node->set_input(edge_a, "input_a");
  sync_node->set_input(edge_b, "input_b");
  sync_node->set_initial_ids({"input_a", "input_b"});
  proc.get_graph()->add_node(sync_node);

  auto add_node = std::make_shared<test_add_node>();
  add_node->set_input(sync_node->get_output(), "default");
  proc.get_graph()->add_node(add_node);

  bool received = false;
  std::shared_ptr<result_message> result_msg;

  auto output_edge = add_node->get_output("default");
  output_edge->set_callback(std::make_shared<graph_message_callback>([&](graph_message_ptr msg) {
    result_msg = std::dynamic_pointer_cast<result_message>(msg);
    received = true;
  }));

  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> data_b = {10.0f, 20.0f, 30.0f, 40.0f};
  input_node->add_frame(1, {{"input_a", dynamic_tensor(dtype::float32, {2, 2}, data_a.data())},
                            {"input_b", dynamic_tensor(dtype::float32, {2, 2}, data_b.data())}});

  proc.run();

  ASSERT_TRUE(received);
  ASSERT_TRUE(result_msg);
  ASSERT_TRUE(result_msg->is_ok());

  auto tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(result_msg->get_value());
  ASSERT_TRUE(tensor_msg);

  const auto& result = tensor_msg->get_tensor();
  EXPECT_EQ(result.shape(), std::vector<int64_t>({2, 2}));

  const float* result_data = result.data_ptr<float>();
  std::vector<float> expected = {11.0f, 22.0f, 33.0f, 44.0f};

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result_data[i], expected[i], 1e-5);
  }
}

TEST(FrameSyncTest, MultipleFrames) {
  graph_proc proc;

  auto input_node = std::make_shared<test_input_node>();
  proc.get_graph()->add_node(input_node);

  auto extractor = std::make_shared<result_field_extractor_node>();
  extractor->set_input(input_node->get_output());
  proc.get_graph()->add_node(extractor);

  auto edge_a = extractor->add_output("input_a");
  auto edge_b = extractor->add_output("input_b");

  auto sync_node = std::make_shared<result_message_sync_node>();
  sync_node->set_input(edge_a, "input_a");
  sync_node->set_input(edge_b, "input_b");
  sync_node->set_initial_ids({"input_a", "input_b"});
  proc.get_graph()->add_node(sync_node);

  auto add_node = std::make_shared<test_add_node>();
  add_node->set_input(sync_node->get_output(), "default");
  proc.get_graph()->add_node(add_node);

  std::vector<std::pair<uint64_t, dynamic_tensor>> results;

  auto output_edge = add_node->get_output("default");
  output_edge->set_callback(std::make_shared<graph_message_callback>([&](graph_message_ptr msg) {
    if (auto result_msg = std::dynamic_pointer_cast<result_message>(msg)) {
      if (result_msg->is_ok()) {
        if (auto tensor_msg =
                std::dynamic_pointer_cast<dynamic_tensor_message>(result_msg->get_value())) {
          results.push_back({result_msg->get_frame_number(), tensor_msg->get_tensor()});
        }
      }
    }
  }));

  std::vector<float> data1_a = {1.0f};
  std::vector<float> data1_b = {10.0f};
  input_node->add_frame(1, {{"input_a", dynamic_tensor(dtype::float32, {1}, data1_a.data())},
                            {"input_b", dynamic_tensor(dtype::float32, {1}, data1_b.data())}});

  std::vector<float> data2_a = {2.0f};
  std::vector<float> data2_b = {20.0f};
  input_node->add_frame(2, {{"input_a", dynamic_tensor(dtype::float32, {1}, data2_a.data())},
                            {"input_b", dynamic_tensor(dtype::float32, {1}, data2_b.data())}});

  proc.run();

  ASSERT_EQ(results.size(), 2);

  EXPECT_EQ(results[0].first, 1);
  float result1 = results[0].second.data_ptr<float>()[0];
  EXPECT_NEAR(result1, 11.0f, 1e-5);

  EXPECT_EQ(results[1].first, 2);
  float result2 = results[1].second.data_ptr<float>()[0];
  EXPECT_NEAR(result2, 22.0f, 1e-5);
}

TEST(FrameSyncTest, ThreeWaySync) {
  graph_proc proc;

  auto input_node = std::make_shared<test_input_node>();
  proc.get_graph()->add_node(input_node);

  std::vector<float> data_a = {1.0f};
  std::vector<float> data_b = {2.0f};
  std::vector<float> data_c = {3.0f};
  input_node->add_frame(1, {{"field_a", dynamic_tensor(dtype::float32, {1}, data_a.data())},
                            {"field_b", dynamic_tensor(dtype::float32, {1}, data_b.data())},
                            {"field_c", dynamic_tensor(dtype::float32, {1}, data_c.data())}});

  auto extractor = std::make_shared<result_field_extractor_node>();
  extractor->set_input(input_node->get_output());
  auto edge_a = extractor->add_output("field_a");
  auto edge_b = extractor->add_output("field_b");
  auto edge_c = extractor->add_output("field_c");
  proc.get_graph()->add_node(extractor);

  auto sync_node = std::make_shared<result_message_sync_node>();
  sync_node->set_input(edge_a, "field_a");
  sync_node->set_input(edge_b, "field_b");
  sync_node->set_input(edge_c, "field_c");
  sync_node->set_initial_ids({"field_a", "field_b", "field_c"});
  proc.get_graph()->add_node(sync_node);

  bool received = false;
  std::shared_ptr<result_message> result_msg;

  auto output_edge = sync_node->get_output();
  output_edge->set_callback(std::make_shared<graph_message_callback>([&](graph_message_ptr msg) {
    result_msg = std::dynamic_pointer_cast<result_message>(msg);
    received = true;
  }));

  proc.run();

  ASSERT_TRUE(received);

  auto field_a = result_msg->get_field("field_a");
  auto field_b = result_msg->get_field("field_b");
  auto field_c = result_msg->get_field("field_c");

  ASSERT_TRUE(field_a);
  ASSERT_TRUE(field_b);
  ASSERT_TRUE(field_c);
}

TEST(FrameSyncTest, MultiStageSync) {
  graph_proc proc;

  auto input_node = std::make_shared<test_input_node>();
  proc.get_graph()->add_node(input_node);

  std::vector<float> data_a = {1.0f};
  std::vector<float> data_b = {2.0f};
  std::vector<float> data_c = {3.0f};
  input_node->add_frame(1, {{"data_a", dynamic_tensor(dtype::float32, {1}, data_a.data())},
                            {"data_b", dynamic_tensor(dtype::float32, {1}, data_b.data())},
                            {"data_c", dynamic_tensor(dtype::float32, {1}, data_c.data())}});

  auto extractor = std::make_shared<result_field_extractor_node>();
  extractor->set_input(input_node->get_output());
  auto edge_a = extractor->add_output("data_a");
  auto edge_b = extractor->add_output("data_b");
  auto edge_c = extractor->add_output("data_c");
  proc.get_graph()->add_node(extractor);

  auto sync_node_1 = std::make_shared<result_message_sync_node>();
  sync_node_1->set_input(edge_a, "data_a");
  sync_node_1->set_input(edge_b, "data_b");
  sync_node_1->set_initial_ids({"data_a", "data_b"});
  proc.get_graph()->add_node(sync_node_1);

  auto add_node_1 = std::make_shared<test_add_node>();
  add_node_1->set_input_a_name("data_a");
  add_node_1->set_input_b_name("data_b");
  add_node_1->set_input(sync_node_1->get_output(), "default");
  proc.get_graph()->add_node(add_node_1);

  auto sync_node_2 = std::make_shared<result_message_sync_node>();
  sync_node_2->set_input(add_node_1->get_output("default"), "output");
  sync_node_2->set_input(edge_c, "data_c");
  sync_node_2->set_initial_ids({"output", "data_c"});
  proc.get_graph()->add_node(sync_node_2);

  auto add_node_2 = std::make_shared<test_add_node>();
  add_node_2->set_input_a_name("output");
  add_node_2->set_input_b_name("data_c");
  add_node_2->set_input(sync_node_2->get_output(), "default");
  proc.get_graph()->add_node(add_node_2);

  bool received = false;
  std::shared_ptr<result_message> result_msg;

  add_node_2->get_output("default")->set_callback(
      std::make_shared<graph_message_callback>([&](graph_message_ptr msg) {
        if (auto res_msg = std::dynamic_pointer_cast<result_message>(msg)) {
          result_msg = res_msg;
          received = true;
        }
      }));

  proc.run();

  ASSERT_TRUE(received);
  ASSERT_TRUE(result_msg->is_ok());

  auto tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(result_msg->get_value());
  ASSERT_TRUE(tensor_msg);

  float result = tensor_msg->get_tensor().data_ptr<float>()[0];
  EXPECT_NEAR(result, 6.0f, 1e-5);
}

TEST(FrameSyncTest, ErrorPropagation) {
  class error_generator_node : public graph_node {
    graph_edge_ptr output_;

   public:
    error_generator_node() : graph_node(), output_(std::make_shared<graph_edge>(this)) {
      set_output(output_);
    }
    std::string get_proc_name() const override { return "error_generator_node"; }
    void process(std::string input_name, graph_message_ptr message) override {
      if (auto result_msg = std::dynamic_pointer_cast<result_message>(message)) {
        std::unordered_map<std::string, graph_message_ptr> fields;
        const auto& input_fields = result_msg->get_fields();
        for (const auto& [name, value] : input_fields) {
          fields[name] = nullptr;
        }
        auto error = result_message::error(fields, "Processing error");
        error->set_frame_number(result_msg->get_frame_number());
        output_->send(error);
      }
    }
  };

  graph_proc proc;

  auto input_node = std::make_shared<test_input_node>();
  proc.get_graph()->add_node(input_node);

  std::vector<float> data_a = {1.0f};
  std::vector<float> data_b = {2.0f};
  input_node->add_frame(1, {{"input_a", dynamic_tensor(dtype::float32, {1}, data_a.data())},
                            {"input_b", dynamic_tensor(dtype::float32, {1}, data_b.data())}});

  auto extractor = std::make_shared<result_field_extractor_node>();
  extractor->set_input(input_node->get_output());
  auto edge_a = extractor->add_output("input_a");
  auto edge_b = extractor->add_output("input_b");
  proc.get_graph()->add_node(extractor);

  auto error_gen = std::make_shared<error_generator_node>();
  error_gen->set_input(edge_a, "default");
  proc.get_graph()->add_node(error_gen);

  auto sync_node = std::make_shared<result_message_sync_node>();
  sync_node->set_input(error_gen->get_output(), "input_a");
  sync_node->set_input(edge_b, "input_b");
  sync_node->set_initial_ids({"input_a", "input_b"});
  proc.get_graph()->add_node(sync_node);

  bool error_received = false;
  auto output_edge = sync_node->get_output();
  output_edge->set_callback(std::make_shared<graph_message_callback>([&](graph_message_ptr msg) {
    if (auto result_msg = std::dynamic_pointer_cast<result_message>(msg)) {
      if (!result_msg->is_ok()) {
        error_received = true;
      }
    }
  }));

  proc.run();

  ASSERT_TRUE(error_received);
}
