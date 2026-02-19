#pragma once

#include <atomic>
#include <cereal/types/base_class.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "coalsack/core/graph_node.h"
#include "coalsack/messages/text_message.h"

namespace coalsack {

class heartbeat_node : public graph_node {
  uint32_t interval;
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;

 public:
  heartbeat_node() : graph_node(), interval(1000), running(false) {}

  virtual std::string get_proc_name() const override { return "heartbeat"; }

  void set_interval(uint32_t interval) { this->interval = interval; }

  uint32_t get_interval() const { return interval; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(interval);
  }

  virtual void run() override {
    running = true;
    th.reset(new std::thread([this]() {
      while (running.load()) {
        tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
      }
    }));
  }

  virtual void stop() override {
    if (running.load()) {
      running.store(false);
      if (th && th->joinable()) {
        th->join();
      }
    }
  }

  virtual void tick() {}
};

class text_heartbeat_node : public heartbeat_node {
  std::string message;
  graph_edge_ptr output;

 public:
  text_heartbeat_node() : heartbeat_node(), message(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  void set_message(std::string message) { this->message = message; }

  std::string get_message() const { return message; }

  virtual std::string get_proc_name() const override { return "text_heartbeat"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<heartbeat_node>(this));
    archive(message);
  }

  virtual void tick() override {
    auto msg = std::make_shared<text_message>();
    msg->set_text(message);
    output->send(msg);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::heartbeat_node, coalsack::graph_node)
COALSACK_REGISTER_NODE(coalsack::text_heartbeat_node, coalsack::heartbeat_node)
