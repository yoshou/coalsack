#pragma once

#include "../nn_op_node.h"

namespace coalsack {

class gather_node : public graph_node {
 protected:
  graph_edge_ptr output_;
  int64_t axis_;
  std::string output_name_;
  std::string data_name_;
  std::string indices_name_;
  std::string node_name_;

 public:
  gather_node()
      : graph_node(), output_(std::make_shared<graph_edge>(this)), axis_(0), output_name_() {
    set_output(output_);
  }

  virtual std::string get_proc_name() const override { return "nn_gather"; }

  void set_axis(int64_t axis) { axis_ = axis; }

  void set_output_name(const std::string& name) { output_name_ = name; }

  void set_node_name(const std::string& name) { node_name_ = name; }

  void set_input_names(const std::string& data_name, const std::string& indices_name) {
    data_name_ = data_name;
    indices_name_ = indices_name;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    uint64_t frame_number = 0;
    auto result_msg = std::dynamic_pointer_cast<result_message>(message);
    if (result_msg) {
      frame_number = result_msg->get_frame_number();
    }

    if (result_msg && !result_msg->is_ok()) {
      spdlog::trace("Skipping gather [{}] (Frame: {})", node_name_, frame_number);
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      auto output_msg = result_message::error(fields, "Input error");
      output_msg->set_frame_number(frame_number);
      output_->send(output_msg);
      return;
    }

    spdlog::trace("Executing gather [{}] (Frame: {})", node_name_, frame_number);

    std::shared_ptr<result_message> output_msg;
    try {
      if (data_name_.empty() || indices_name_.empty()) {
        throw std::runtime_error("gather: input names not set");
      }

      dynamic_tensor data = get_tensor_from_result_message(result_msg, data_name_);
      dynamic_tensor indices = get_tensor_from_result_message(result_msg, indices_name_);

      log_node_input("gather", node_name_, 0, data);
      log_node_input("gather", node_name_, 1, indices);

      const auto& data_shape = data.shape();
      const auto& indices_shape = indices.shape();

      int64_t axis = axis_;
      if (axis < 0) axis += data_shape.size();
      if (axis < 0 || axis >= static_cast<int64_t>(data_shape.size())) {
        throw std::runtime_error("gather: axis out of range");
      }

      std::vector<int64_t> out_shape;
      for (int64_t i = 0; i < axis; ++i) {
        out_shape.push_back(data_shape[i]);
      }
      for (auto dim : indices_shape) {
        out_shape.push_back(dim);
      }
      for (int64_t i = axis + 1; i < static_cast<int64_t>(data_shape.size()); ++i) {
        out_shape.push_back(data_shape[i]);
      }

      dynamic_tensor output(data.get_dtype(), out_shape);

      int64_t outer_size = 1;
      for (int64_t i = 0; i < axis; ++i) {
        outer_size *= data_shape[i];
      }

      int64_t inner_size = 1;
      for (int64_t i = axis + 1; i < static_cast<int64_t>(data_shape.size()); ++i) {
        inner_size *= data_shape[i];
      }

      int64_t indices_size = indices.numel();

      if (data.get_dtype() == dtype::float32 && indices.get_dtype() == dtype::int32) {
        gather_impl<float, int32_t>(data, indices, output, outer_size, indices_size, inner_size,
                                    axis, data_shape);
      } else if (data.get_dtype() == dtype::float32 && indices.get_dtype() == dtype::int64) {
        gather_impl<float, int64_t>(data, indices, output, outer_size, indices_size, inner_size,
                                    axis, data_shape);
      } else if (data.get_dtype() == dtype::float64 && indices.get_dtype() == dtype::int32) {
        gather_impl<double, int32_t>(data, indices, output, outer_size, indices_size, inner_size,
                                     axis, data_shape);
      } else if (data.get_dtype() == dtype::float64 && indices.get_dtype() == dtype::int64) {
        gather_impl<double, int64_t>(data, indices, output, outer_size, indices_size, inner_size,
                                     axis, data_shape);
      } else if (data.get_dtype() == dtype::int32 && indices.get_dtype() == dtype::int32) {
        gather_impl<int32_t, int32_t>(data, indices, output, outer_size, indices_size, inner_size,
                                      axis, data_shape);
      } else if (data.get_dtype() == dtype::int32 && indices.get_dtype() == dtype::int64) {
        gather_impl<int32_t, int64_t>(data, indices, output, outer_size, indices_size, inner_size,
                                      axis, data_shape);
      } else if (data.get_dtype() == dtype::int64 && indices.get_dtype() == dtype::int32) {
        gather_impl<int64_t, int32_t>(data, indices, output, outer_size, indices_size, inner_size,
                                      axis, data_shape);
      } else if (data.get_dtype() == dtype::int64 && indices.get_dtype() == dtype::int64) {
        gather_impl<int64_t, int64_t>(data, indices, output, outer_size, indices_size, inner_size,
                                      axis, data_shape);
      } else {
        throw std::runtime_error("gather: unsupported data/indices type combination");
      }

      log_node_output("gather", node_name_, output);

      auto output_tensor_msg = std::make_shared<dynamic_tensor_message>(output);

      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = output_tensor_msg;

      output_msg = result_message::ok(fields);
      output_msg->set_frame_number(frame_number);
    } catch (const std::exception& e) {
      spdlog::error("gather [{}]: {}", node_name_, e.what());
      std::unordered_map<std::string, graph_message_ptr> fields;
      fields[output_name_] = nullptr;
      output_msg = result_message::error(fields, e.what());
      output_msg->set_frame_number(frame_number);
    }
    output_->send(output_msg);
  }

 private:
  template <typename T, typename Tind>
  void gather_impl(const dynamic_tensor& data, const dynamic_tensor& indices,
                   dynamic_tensor& output, int64_t outer_size, int64_t indices_size,
                   int64_t inner_size, int64_t axis, const std::vector<int64_t>& data_shape) {
    const T* data_ptr = data.data_ptr<T>();
    const Tind* indices_ptr = indices.data_ptr<Tind>();
    T* out_ptr = output.data_ptr<T>();

    for (int64_t outer = 0; outer < outer_size; ++outer) {
      for (int64_t idx = 0; idx < indices_size; ++idx) {
        int64_t gather_idx = static_cast<int64_t>(indices_ptr[idx]);
        if (gather_idx < 0) gather_idx += data_shape[axis];

        if (gather_idx < 0 || gather_idx >= data_shape[axis]) {
          throw std::runtime_error("gather: index out of range");
        }

        int64_t src_offset = outer * data_shape[axis] * inner_size + gather_idx * inner_size;
        int64_t dst_offset = outer * indices_size * inner_size + idx * inner_size;

        std::memcpy(out_ptr + dst_offset, data_ptr + src_offset, inner_size * sizeof(T));
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::gather_node, coalsack::graph_node)
