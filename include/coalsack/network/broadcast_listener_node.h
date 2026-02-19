#pragma once

#include <cereal/archives/binary.hpp>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include "coalsack/core/graph_node.h"
#include "coalsack/data_stream/data_stream_receiver.h"

namespace coalsack {

class broadcast_listener_node : public graph_node {
  graph_edge_ptr output;
  std::shared_ptr<data_stream_receiver> receiver;
  std::string address;
  uint16_t port;
  std::string multicast_address;

 public:
  broadcast_listener_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "broadcast_listener"; }

  void set_endpoint(std::string address, uint16_t port,
                    std::string multicast_address = std::string()) {
    this->address = address;
    this->port = port;
    this->multicast_address = multicast_address;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(address);
    archive(port);
    archive(multicast_address);
  }

  virtual void initialize() override {
    if (!multicast_address.empty()) {
      receiver.reset(new data_stream_receiver(udp::endpoint(udp::v4(), port), multicast_address));
    } else {
      receiver.reset(new data_stream_receiver(udp::endpoint(udp::v4(), port), true));
    }

    source_identifier id{0, 0};
    receiver->add_session(id);
  }

  virtual void run() override {
    receiver->start([this](double timestamp, source_identifier id, asio::streambuf& stream) {
      this->on_receive_data_handler(stream);
    });
  }

  virtual void stop() override { receiver->stop(); }

  void on_receive_data_handler(asio::streambuf& stream) {
    if (stream.size() < sizeof(int)) {
      return;
    }

    std::string str(boost::asio::buffers_begin(stream.data()),
                    boost::asio::buffers_end(stream.data()));
    std::stringstream ss(str);

    std::shared_ptr<graph_message> msg;
    {
      cereal::BinaryInputArchive iarchive(ss);
      iarchive(msg);
    }

    output->send(msg);
  }
};

}  // namespace coalsack

#include "coalsack/core/graph_proc_registry.h"

COALSACK_REGISTER_NODE(coalsack::broadcast_listener_node, coalsack::graph_node)
