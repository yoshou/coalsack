#pragma once

#include "dynamic_mx_tensor.h"
#include "graph_proc.h"

namespace coalsack {

class dynamic_mx_tensor_message : public graph_message {
 private:
  dynamic_mx_tensor tensor_;

 public:
  dynamic_mx_tensor_message() : graph_message(), tensor_() {}
  
  explicit dynamic_mx_tensor_message(const dynamic_mx_tensor& tensor)
      : graph_message(), tensor_(tensor) {}

  const dynamic_mx_tensor& get_mx_tensor() const { return tensor_; }
  dynamic_mx_tensor& get_mx_tensor() { return tensor_; }
  
  void set_mx_tensor(const dynamic_mx_tensor& tensor) { tensor_ = tensor; }

  template <typename Archive>
  void serialize(Archive& archive) {
    graph_message::serialize(archive);
  }
};

}  // namespace coalsack

COALSACK_REGISTER_MESSAGE(coalsack::dynamic_mx_tensor_message, coalsack::graph_message)
