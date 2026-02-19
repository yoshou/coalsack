#pragma once

#include <spdlog/spdlog.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "coalsack/core/graph_node.h"

namespace coalsack {

class fifo_node : public graph_node {
  graph_edge_ptr output;
  std::mutex mtx;
  std::deque<graph_message_ptr> messages;
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;
  std::condition_variable cv;
  std::uint32_t max_size;

 public:
  fifo_node() : graph_node(), output(std::make_shared<graph_edge>(this)), max_size(10) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "fifo"; }

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
        std::cout << "Fifo overflow" << std::endl;
        spdlog::error("Fifo overflow");
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

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::fifo_node, coalsack::graph_node)
