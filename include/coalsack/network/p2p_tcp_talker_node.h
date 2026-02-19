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
#include "coalsack/util/utils.h"

namespace coalsack {

class p2p_tcp_talker_node : public graph_node {
  graph_edge_ptr output;
  std::shared_ptr<data_stream_tcp_transmitter> transmitter;
  boost::asio::io_context io_context;
  std::shared_ptr<std::thread> th;
  std::atomic_bool running;

 public:
  p2p_tcp_talker_node()
      : graph_node(), output(std::make_shared<graph_edge>(this, EDGE_TYPE::CHAIN)), transmitter() {
    set_output(output);
  }

  virtual ~p2p_tcp_talker_node() { finalize(); }

  virtual void finalize() override {
    if (transmitter) {
      transmitter->close();
      transmitter.reset();
    }
  }

  virtual void initialize() override {
    const auto output_req = get_output()->request;
    const auto data = output_req.get_data();
    std::stringstream ss(std::string(data.begin(), data.end()));

    const auto address = read_string(ss);
    const auto port = read_uint16(ss);

    transmitter.reset(new data_stream_tcp_transmitter(io_context));
    transmitter->open(address, port);
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

  virtual std::string get_proc_name() const override { return "data_tcp_talker"; }

  template <typename Archive>
  void serialize(Archive& archive) {}

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

COALSACK_REGISTER_NODE(coalsack::p2p_tcp_talker_node, coalsack::graph_node)
