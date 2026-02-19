#pragma once

#include <atomic>
#include <cereal/archives/binary.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "coalsack/core/graph_node.h"
#include "coalsack/data_stream/data_stream_transmitter.h"

namespace coalsack {

class broadcast_talker_node : public graph_node {
  std::shared_ptr<data_stream_transmitter> transmitter;
  boost::asio::io_context io_context;
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;
  std::string address;
  std::uint16_t port;

 public:
  broadcast_talker_node() : graph_node(), transmitter(), io_context(), th(), running(false) {}

  void set_endpoint(std::string address, uint16_t port) {
    this->address = address;
    this->port = port;
  }

  virtual ~broadcast_talker_node() { finalize(); }

  virtual void finalize() override {
    if (transmitter) {
      transmitter->close();
      transmitter.reset();
    }
  }

  virtual void initialize() override {
    transmitter.reset(new data_stream_transmitter(io_context));

    if (asio::ip::make_address(address).is_multicast()) {
      transmitter->open(address, port);
    } else {
      transmitter->open_broadcast(port);
    }
  }

  virtual void process([[maybe_unused]] std::string input_name,
                       graph_message_ptr message) override {
    source_identifier id{0, 0};
    std::stringstream ss;
    {
      cereal::BinaryOutputArchive oarchive(ss);
      oarchive(message);
    }
    std::string str = ss.str();

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();

    transmitter->send(id, (double)(ns / 100), (uint8_t*)str.data(), str.size());
  }

  virtual std::string get_proc_name() const override { return "broadcast_talker"; }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(address);
    archive(port);
  }

  virtual void run() override {
    running = true;
    th.reset(new std::thread([&]() { io_context.run(); }));
  }

  virtual void stop() override {
    if (running.load()) {
      running.store(false);
      io_context.stop();
      if (th && th->joinable()) {
        th->join();
      }
    }
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::broadcast_talker_node, coalsack::graph_node)
