/// @file p2p_tcp_listener_node.h
/// @brief TCP point-to-point message receiver node (reliable delivery).
/// @ingroup network_nodes
#pragma once

#include <cereal/archives/binary.hpp>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "coalsack/core/graph_node.h"
#include "coalsack/data_stream/data_stream_receiver.h"
#include "coalsack/messages/text_message.h"
#include "coalsack/util/utils.h"

namespace coalsack {

/// @brief Listens for incoming TCP connections and deserializes each message into a @c graph_message.
/// @details Accepts one TCP connection, reads length-prefixed frames, Cereal-deserializes each,
///          and forwards the result on @b "default".
///
/// @par Inputs
///   (none — autonomous network source)
///
/// @par Outputs
/// - @b "default" — deserialized @c graph_message
///
/// @par Properties
/// - address (std::string) — local bind address
/// - port    (uint16_t)    — local TCP port to listen on
/// @see p2p_tcp_talker_node, p2p_listener_node
class p2p_tcp_listener_node : public graph_node {
  graph_edge_ptr output;
  std::shared_ptr<data_stream_tcp_receiver> receiver;
  std::string address;
  uint16_t port;

 public:
  p2p_tcp_listener_node() : graph_node(), output(std::make_shared<graph_edge>(this)) {
    set_output(output);
  }

  virtual std::string get_proc_name() const override { return "data_tcp_listener"; }

  void set_endpoint(std::string address, uint16_t port) {
    this->address = address;
    this->port = port;
  }

  template <typename Archive>
  void serialize(Archive& archive) {
    archive(address);
    archive(port);
  }

  virtual void initialize() override {
    receiver.reset(
        new data_stream_tcp_receiver(tcp::endpoint(asio::ip::make_address(address), port)));

    const source_identifier id{0, 0};
    receiver->add_session(id);

    const auto bind_port = receiver->local_endpoint().port();
    const auto bind_address = receiver->local_endpoint().address().to_string();

    std::stringstream ss;
    write_string(ss, address);
    write_uint16(ss, bind_port);

    const auto str = ss.str();
    const std::vector<uint8_t> data(str.begin(), str.end());

    subscribe_request req;
    const auto input = get_input();
    req.set_proc_name(get_proc_name());
    req.set_msg_type(text_message::get_type());
    req.set_data(data);
    input->request = req;
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

COALSACK_REGISTER_NODE(coalsack::p2p_tcp_listener_node, coalsack::graph_node)
