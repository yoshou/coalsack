#pragma once

#include <malloc.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "dynamic_tensor.h"
#include "dynamic_tensor_message.h"
#include "graph_proc.h"
#include "result_message.h"

namespace coalsack {

class model_input_node : public graph_node {
 private:
  std::unordered_map<std::string, dynamic_tensor> tensors_;
  graph_edge_ptr output_;
  uint64_t frame_number_;

 public:
  model_input_node() : graph_node(), output_(std::make_shared<graph_edge>(this)), frame_number_(1) {
    set_output(output_);
  }

  virtual std::string get_proc_name() const override { return "model_input"; }

  void set_tensor(const std::string& name, const dynamic_tensor& tensor) {
    tensors_[name] = tensor;
  }
  void set_frame_number(uint64_t fn) { frame_number_ = fn; }
  graph_edge_ptr get_output() const { return output_; }

  virtual void run() override {
    std::shared_ptr<result_message> result;
    try {
      std::unordered_map<std::string, graph_message_ptr> fields;

      for (const auto& [name, tensor] : tensors_) {
        auto tensor_msg = std::make_shared<dynamic_tensor_message>(tensor);
        fields[name] = tensor_msg;
      }

      result = result_message::ok(fields);
      result->set_frame_number(frame_number_);

    } catch (const std::exception& e) {
      std::unordered_map<std::string, graph_message_ptr> fields;
      for (const auto& [name, tensor] : tensors_) {
        fields[name] = nullptr;
      }
      result = result_message::error(fields, e.what());
      result->set_frame_number(frame_number_);
    }
    output_->send(result);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {}
};

class model_source_node : public graph_node {
 private:
  graph_edge_ptr output_;
  std::mutex mtx_;
  std::deque<graph_message_ptr> queue_;
  std::thread worker_;
  std::atomic<bool> running_;
  std::condition_variable cv_;
  std::uint32_t max_size_;

  std::unordered_map<std::string, dynamic_tensor> pending_tensors_;
  uint64_t pending_frame_;

 public:
  model_source_node()
      : graph_node(),
        output_(std::make_shared<graph_edge>(this)),
        running_(false),
        max_size_(10),
        pending_frame_(1) {
    set_output(output_);
  }

  virtual ~model_source_node() {
    stop();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  virtual std::string get_proc_name() const override { return "model_source"; }

  void set_tensor(const std::string& name, const dynamic_tensor& tensor) {
    std::lock_guard<std::mutex> lock(mtx_);
    pending_tensors_[name] = tensor;
  }

  void set_frame_number(uint64_t fn) {
    std::lock_guard<std::mutex> lock(mtx_);
    pending_frame_ = fn;
  }

  void push() {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!running_) {
      return;
    }

    std::shared_ptr<result_message> result;
    try {
      std::unordered_map<std::string, graph_message_ptr> fields;
      for (const auto& [name, tensor] : pending_tensors_) {
        auto tensor_msg = std::make_shared<dynamic_tensor_message>(tensor);
        fields[name] = tensor_msg;
      }
      result = result_message::ok(fields);
      result->set_frame_number(pending_frame_);
    } catch (const std::exception& e) {
      std::unordered_map<std::string, graph_message_ptr> fields;
      for (const auto& [name, tensor] : pending_tensors_) {
        fields[name] = nullptr;
      }
      result = result_message::error(fields, e.what());
      result->set_frame_number(pending_frame_);
    }

    if (queue_.size() >= max_size_) {
      spdlog::error("model_source_node queue overflow");
    } else {
      queue_.push_back(result);
      cv_.notify_one();
    }
  }

  graph_edge_ptr get_output() const { return output_; }

  virtual void run() override {
    running_ = true;
    worker_ = std::thread([this]() {
      while (running_.load()) {
        graph_message_ptr message;
        {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [&] { return !queue_.empty() || !running_; });

          if (!running_) {
            break;
          }

          if (!queue_.empty()) {
            message = queue_.front();
            queue_.pop_front();
          }
        }

        if (message) {
          output_->send(message);
        }
      }
    });
  }

  virtual void stop() override {
    running_ = false;
    cv_.notify_all();
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {}
};

class model_output_node : public graph_node {
 private:
  std::unordered_map<std::string, dynamic_tensor> outputs_;
  std::function<void(const std::unordered_map<std::string, dynamic_tensor>&)> callback_;

 public:
  model_output_node() : graph_node(), outputs_(), callback_(nullptr) {}

  virtual std::string get_proc_name() const override { return "model_output"; }

  void set_callback(
      std::function<void(const std::unordered_map<std::string, dynamic_tensor>&)> cb) {
    callback_ = cb;
  }

  const std::unordered_map<std::string, dynamic_tensor>& get_collected_outputs() const {
    return outputs_;
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    try {
      auto result_msg = std::dynamic_pointer_cast<result_message>(message);
      if (!result_msg) {
        throw std::runtime_error("Expected result_message");
      }

      if (result_msg->is_error()) {
        throw std::runtime_error("Model output received error: " + result_msg->get_error_message());
      }

      outputs_.clear();
      for (const auto& [name, field_msg] : result_msg->get_fields()) {
        auto tensor_msg = std::dynamic_pointer_cast<dynamic_tensor_message>(field_msg);
        if (tensor_msg) {
          outputs_[name] = tensor_msg->get_tensor();
        }
      }

      if (callback_) {
        callback_(outputs_);
      }

    } catch (const std::exception& e) {
    }
  }
};

class layer_scheduler_node : public graph_node {
  graph_edge_ptr output;
  std::mutex mtx;
  std::deque<graph_message_ptr> messages;
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;
  std::condition_variable cv;
  std::uint32_t max_size;

 public:
  layer_scheduler_node() : graph_node(), output(std::make_shared<graph_edge>(this)), max_size(10) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "layer_scheduler"; }

  void set_max_size(std::uint32_t value) { max_size = value; }
  std::uint32_t get_max_size() const { return max_size; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(max_size);
  }

  virtual void process(std::string input_name, graph_message_ptr message) override {
    if (!running) {
      return;
    }

    if (input_name == "default") {
      std::lock_guard<std::mutex> lock(mtx);

      if (messages.size() >= max_size) {
        std::cout << "Layer scheduler overflow" << std::endl;
        spdlog::error("Layer scheduler overflow");
      } else {
        messages.push_back(message);
        cv.notify_one();
      }
    }
  }

  virtual void run() override {
    running = true;
    th.reset(new std::thread([this]() {
      while (running.load()) {
        graph_message_ptr message;
        {
          std::unique_lock<std::mutex> lock(mtx);
          cv.wait(lock, [&] { return !messages.empty() || !running; });

          if (!running) {
            break;
          }

          message = messages.front();
          messages.pop_front();
        }

        if (message) {
          output->send(message);

          // Trim fragmented memory after layer processing
          malloc_trim(0);
        }
      }
    }));
  }

  virtual void stop() override {
    if (running.load()) {
      {
        std::lock_guard<std::mutex> lock(mtx);
        running = false;
      }
      cv.notify_one();
      if (th && th->joinable()) {
        th->join();
      }
    }
  }
};

}  // namespace coalsack

COALSACK_REGISTER_NODE(coalsack::model_input_node, coalsack::graph_node)
COALSACK_REGISTER_NODE(coalsack::model_output_node, coalsack::graph_node)
COALSACK_REGISTER_NODE(coalsack::layer_scheduler_node, coalsack::graph_node)
