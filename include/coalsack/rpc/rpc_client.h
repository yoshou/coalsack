#pragma once

// clang-format off
#include <utility>

#include <boost/array.hpp>
#include <boost/asio.hpp>
// clang-format on

#include "coalsack/rpc/rpc_common.h"

namespace coalsack {
namespace asio = boost::asio;
using asio::ip::tcp;
using asio::ip::udp;

class rpc_client {
  tcp::socket socket;

 public:
  tcp::endpoint local_endpoint() const { return socket.local_endpoint(); }

  rpc_client(asio::io_context &io_context) : socket(io_context) {}

  void connect(std::string ip, unsigned short port) {
    boost::system::error_code error;
    socket.connect(tcp::endpoint(asio::ip::make_address(ip), port), error);

    if (error) {
      throw std::runtime_error("Failed connecting to " + ip + ":" + std::to_string(port) + " (" +
                               error.message() + ")");
    }
  }

  int64_t invoke(uint32_t func, const std::vector<uint8_t> &arg, std::vector<uint8_t> &res) {
    boost::system::error_code error;

    assert(arg.size() <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));

    request_t request;
    request.func = func;
    request.id = 0;
    request.length = arg.size();
    asio::write(socket, asio::buffer((const char *)&request, sizeof(request_t)), error);
    asio::write(socket, asio::buffer(arg.data(), arg.size()), error);

    asio::streambuf receive_buffer;
    asio::read(socket, receive_buffer, asio::transfer_exactly(sizeof(response_t)), error);

    if (error) {
      return -1;
    }

    response_t response = *static_cast<const response_t *>(receive_buffer.data().data());
    receive_buffer.consume(sizeof(response_t));

    if (response.length > 0) {
      asio::read(socket, receive_buffer, asio::transfer_exactly(response.length), error);

      const char *data = static_cast<const char *>(receive_buffer.data().data());
      std::copy(data, data + response.length, std::back_inserter(res));
      receive_buffer.consume(response.length);
    }

    return (int64_t)response.code;
  }
};
}  // namespace coalsack
