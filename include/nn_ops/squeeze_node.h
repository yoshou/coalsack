#pragma once

#include <set>

#include "../nn_op_node.h"

namespace coalsack {

class squeeze_node : public graph_node {
 protected:
  graph_edge_ptr output_;
  std::vector<int64_t> axes_;
  std::string output_name_;
  std::string data_name_;
  std::string axes_input_name_;
  std::string node_name_;

 public:
  squeeze_node() : graph_node(), output_(std::make_shared<graph_edge>(this)), output_name_() {
    set_output(output_);
  }

  void set_axes(const std::vector<int64_t>& axes) { axes_ = axes; }

  void set_input_names(const std::string& data_name, const std::string& axes_name = "") {
    data_name_ = data_name;
    axes_input_name_ = axes_name;
  }

  void set_output_name(const std::string& name) { output_name_ = name; }

  void set_node_name(const std::string& name) { node_name_ = name; }

  virtual std::string get_proc_name() const override { return "squeeze"; }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (!result_msg) {
      throw std::runtime_error("squeeze: expected result_message");
    }

    uint64_t frame_number = result_msg->get_frame_number();

    if (!result_msg->is_ok()) {
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }

    std::shared_ptr<result_message> output_msg;
    try {
      dynamic_tensor data = get_tensor_from_result_message(result_msg, data_name_);
      log_node_input("squeeze", node_name_, 0, data);

      // Get axes (from input or attribute)
      std::vector<int64_t> axes = axes_;
      if (!axes_input_name_.empty()) {
        auto axes_tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(
            result_msg->get_field(axes_input_name_));
        if (!axes_tensor_msg || axes_tensor_msg->get_tensor().get_dtype() != dtype::int64) {
          throw std::runtime_error("squeeze: invalid axes input");
        }
        const int64_t* axes_data = axes_tensor_msg->get_tensor().data_ptr<int64_t>();
        axes.assign(axes_data, axes_data + axes_tensor_msg->get_tensor().numel());
      }

      const auto& in_shape = data.shape();
      std::vector<int64_t> out_shape;

      if (axes.empty()) {
        for (auto dim : in_shape) {
          if (dim != 1) out_shape.push_back(dim);
        }
      } else {
        std::set<int64_t> axes_set;
        for (auto axis : axes) {
          if (axis < 0) axis += in_shape.size();
          axes_set.insert(axis);
        }

        for (size_t i = 0; i < in_shape.size(); ++i) {
          if (axes_set.count(i)) {
            if (in_shape[i] != 1) {
              throw std::runtime_error("squeeze: cannot squeeze dimension of size != 1");
            }
          } else {
            out_shape.push_back(in_shape[i]);
          }
        }
      }

      // ONNX allows scalar output (empty shape)
      if (out_shape.empty()) {
        out_shape.push_back(1);
      }

      dynamic_tensor output(data.get_dtype(), out_shape);
      std::memcpy(output.data(), data.data(), data.numel() * dtype_size(data.get_dtype()));

      log_node_output("squeeze", node_name_, output);

      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = std::make_shared<dynamic_tensor_message>(output);
      output_msg = result_message::ok(fields);
      output_msg->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("squeeze [{}]: {}", node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      output_msg = result_message::error(fields, e.what());
      output_msg->set_frame_number(frame_number);
    }
    output_->send(output_msg);
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::squeeze_node, coalsack::graph_node)
