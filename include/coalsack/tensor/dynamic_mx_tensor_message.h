/// @file dynamic_mx_tensor_message.h
/// @brief graph_message wrapper for dynamic_mx_tensor.
/// @ingroup tensor
#pragma once

#include "coalsack/core/graph_message.h"
#include "coalsack/core/graph_proc_registry.h"
#include "coalsack/tensor/dynamic_mx_tensor.h"

namespace coalsack {

/// @brief Message transporting an MXFP4 mixed-precision tensor between nodes.
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
