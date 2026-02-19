#pragma once

#include "coalsack/core/graph_message.h"
#include "coalsack/core/graph_proc_registry.h"
#include "coalsack/tensor/dynamic_tensor.h"

namespace coalsack {

class dynamic_tensor_message : public graph_message {
 private:
  dynamic_tensor tensor_;

 public:
  dynamic_tensor_message() : graph_message(), tensor_() {}

  explicit dynamic_tensor_message(const dynamic_tensor& tensor)
      : graph_message(), tensor_(tensor) {}

  dynamic_tensor_message(dtype dt, const std::vector<int64_t>& shape, const void* data = nullptr)
      : graph_message(), tensor_(dt, shape, data) {}

  const dynamic_tensor& get_tensor() const { return tensor_; }
  dynamic_tensor& get_tensor() { return tensor_; }

  void set_tensor(const dynamic_tensor& tensor) { tensor_ = tensor; }

  template <typename Archive>
  void serialize(Archive& archive) {
    graph_message::serialize(archive);
  }
};

}  // namespace coalsack

COALSACK_REGISTER_MESSAGE(coalsack::dynamic_tensor_message, coalsack::graph_message)
