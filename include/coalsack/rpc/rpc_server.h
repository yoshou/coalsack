#pragma once

// clang-format off
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
// clang-format on

#include "coalsack/rpc/rpc_common.h"

namespace coalsack {
namespace asio = boost::asio;
using asio::ip::tcp;

using rpc_disconnect_func = std::function<void(uint32_t)>;

class session : public std::enable_shared_from_this<session> {
 public:
  session(tcp::socket socket, uint32_t session_id,
          const std::unordered_map<uint32_t, rpc_func> &handlers,
          const rpc_disconnect_func disconnect_handler)
      : socket_(std::move(socket)),
        receive_buff_(),
        session_id(session_id),
        handlers_(handlers),
        disconnect_handler(disconnect_handler) {}

  void start() { wait_next_request(); }

 private:
  void wait_next_request() {
    auto self(shared_from_this());
    boost::asio::async_read(socket_, receive_buff_,
                            boost::asio::transfer_exactly(sizeof(request_t)),
                            [this, self](boost::system::error_code ec, std::size_t length) {
                              if (!ec) {
                                self->handle_request(length);
                              }

                              if (ec == boost::asio::error::eof) {
                                if (disconnect_handler) {
                                  disconnect_handler(session_id);
                                }
                              }
                            });
  }

  void handle_request(std::size_t bytes_transferred) {
    auto data = receive_buff_.data();
    auto request = *reinterpret_cast<const request_t *>(&*boost::asio::buffers_begin(data));
    receive_buff_.consume(sizeof(request_t));

    if (request.length > 0) {
      auto self(shared_from_this());
      boost::asio::async_read(
          socket_, receive_buff_, boost::asio::transfer_exactly(request.length),
          [request, this, self](boost::system::error_code ec, std::size_t length) {
            if (!ec) {
              auto bytes = reinterpret_cast<const uint8_t *>(
                  &*boost::asio::buffers_begin(receive_buff_.data()));
              std::vector<uint8_t> data(bytes, bytes + request.length);
              constexpr uint32_t max_consume_once =
                  static_cast<uint32_t>(std::numeric_limits<int>::max());
              for (uint32_t consumed = 0; consumed < request.length;
                   consumed += std::min(request.length, max_consume_once)) {
                receive_buff_.consume(std::min(request.length, max_consume_once));
              }
              self->invoke_func(request, data);
            }

            if (ec == boost::asio::error::eof) {
              if (disconnect_handler) {
                disconnect_handler(session_id);
              }
            }
          });
    } else {
      invoke_func(request, std::vector<uint8_t>());
    }
  }

  void invoke_func(request_t req, std::vector<uint8_t> data) {
    auto it = handlers_.find(req.func);
    if (it != handlers_.end()) {
      rpc_func func = (*it).second;

      std::vector<uint8_t> result_data;
      uint32_t result_code = func(session_id, data, result_data);

      response_t res;
      res.id = req.id;
      res.code = result_code;
      res.length = (uint32_t)result_data.size();

      asio::write(socket_, asio::buffer((const char *)&res, sizeof(response_t)));

      if (result_data.size() > 0) {
        asio::write(socket_, asio::buffer((const char *)result_data.data(), result_data.size()));
      }
    }

    wait_next_request();
  }

  tcp::socket socket_;
  boost::asio::streambuf receive_buff_;
  uint32_t session_id;
  const std::unordered_map<uint32_t, rpc_func> &handlers_;
  const rpc_disconnect_func disconnect_handler;
};

class rpc_server {
 public:
  rpc_server(boost::asio::io_context &io_context, std::string address, uint16_t port)
      : acceptor_(io_context, tcp::endpoint(asio::ip::make_address(address), port)),
        socket_(io_context),
        next_session_id(1),
        disconnect_handler(nullptr) {
    do_accept();
  }

  void register_handler(uint32_t id, rpc_func func) { handlers.emplace(id, func); }

  boost::asio::ip::tcp::endpoint remote_endpoint() const { return socket_.remote_endpoint(); }

  void on_discconect(rpc_disconnect_func handler) { disconnect_handler = handler; }

  boost::asio::ip::tcp::endpoint local_endpoint() const { return acceptor_.local_endpoint(); }

 private:
  void do_accept() {
    acceptor_.async_accept(socket_, [this](boost::system::error_code ec) {
      if (!ec) {
        std::make_shared<session>(std::move(socket_), next_session_id, handlers, disconnect_handler)
            ->start();
        next_session_id++;
      }

      do_accept();
    });
  }

  tcp::acceptor acceptor_;
  tcp::socket socket_;
  std::unordered_map<uint32_t, rpc_func> handlers;
  uint32_t next_session_id;
  rpc_disconnect_func disconnect_handler;
};
}  // namespace coalsack
