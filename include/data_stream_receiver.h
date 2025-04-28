#pragma once

#include <spdlog/spdlog.h>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "data_stream_common.h"

namespace coalsack {
namespace asio = boost::asio;
using asio::ip::tcp;
using asio::ip::udp;

struct packet_data {
  boost::array<char, PACKET_PAYLOAD_SIZE> recv_buf;
  double timestamp = 0;
  source_identifier id = {};
  uint16_t counter = 0;
  uint16_t flags = 0;
  uint16_t size = 0;
};

using packet_data_ptr = std::shared_ptr<packet_data>;

class packet_buffer_pool {
  std::vector<std::shared_ptr<packet_data>> allocated_packets;

  std::mutex free_packets_mtx;
  std::stack<packet_data *> free_packets;

 public:
  packet_buffer_pool(size_t prepare_size = 10000) {
    for (size_t i = 0; i < prepare_size; i++) {
      const auto packet = std::make_shared<packet_data>();
      allocated_packets.push_back(packet);
      free_packets.push(packet.get());
    }
  }

  packet_data *obtain_packet() {
    {
      std::lock_guard<std::mutex> lock(free_packets_mtx);
      if (free_packets.size() > 0) {
        const auto packet = free_packets.top();
        free_packets.pop();

        return packet;
      }
    }

    const auto new_packet = std::make_shared<packet_data>();
    allocated_packets.push_back(new_packet);
    return new_packet.get();
  }

  void release_packets(const std::vector<packet_data *> &packets) {
    std::lock_guard<std::mutex> lock(free_packets_mtx);
    for (auto packet : packets) {
      free_packets.push(packet);
    }
  }
};

class reordering_packet_buffer {
  size_t max_size;
  uint16_t next_counter;

  std::mutex ordered_packets_mtx;
  std::unordered_map<uint16_t, packet_data *> ordered_packets;

 public:
  reordering_packet_buffer(size_t max_size = 10) : max_size(max_size), next_counter(0) {}

  void set_next_counter(uint16_t value) { next_counter = value; }

  uint16_t get_next_counter() const { return next_counter; }

  void commit_packet(packet_data *packet) {
    std::lock_guard<std::mutex> lock(ordered_packets_mtx);
    ordered_packets[packet->counter] = packet;
  }

  bool detect_lost_packet() {
    std::lock_guard<std::mutex> lock(ordered_packets_mtx);

    if (ordered_packets.size() > max_size) {
      return true;
    }
    return false;
  }

  packet_data *receive_packet() {
    std::lock_guard<std::mutex> lock(ordered_packets_mtx);

    if (ordered_packets.find(next_counter) != ordered_packets.end()) {
      auto packet = ordered_packets[next_counter];
      ordered_packets.erase(next_counter);
      next_counter = (uint16_t)(((uint32_t)next_counter + 1) % 65536);
      return packet;
    }

    return nullptr;
  }

  std::vector<packet_data *> reset() {
    std::lock_guard<std::mutex> lock(ordered_packets_mtx);
    std::vector<packet_data *> packets;

    for (const auto &[key, packet] : ordered_packets) {
      packets.push_back(packet);
    }
    ordered_packets.clear();

    return packets;
  }
};

class data_stream_receiver {
  asio::io_context io_context;

  struct session {
    asio::streambuf buffer;
    bool lost_packet;

    session() : lost_packet(true) {}
  };
  typedef std::function<void(double, source_identifier, asio::streambuf &)> on_receive_func;
  udp::socket socket_;
  udp::endpoint remote_endpoint_;
  reordering_packet_buffer reordering;
  packet_buffer_pool packet_pool;
  std::map<source_identifier, std::shared_ptr<session>> sessions;
  on_receive_func on_receive;
  std::atomic_bool running;
  std::shared_ptr<std::thread> handling_thread;
  std::shared_ptr<std::thread> io_thread;
  std::queue<packet_data *> packet_queue;
  std::mutex mtx;
  std::condition_variable cv;

 public:
  data_stream_receiver(udp::endpoint endpoint, bool enable_broadcast = false)
      : io_context(), socket_(io_context, endpoint), running(false), mtx() {
    if (enable_broadcast) {
      socket_.set_option(boost::asio::ip::udp::socket::broadcast(true));
    }
  }

  data_stream_receiver(udp::endpoint endpoint, std::string multicast_address)
      : io_context(), socket_(io_context), running(false), mtx() {
    socket_.open(endpoint.protocol());
    socket_.set_option(boost::asio::ip::udp::socket::reuse_address(true));
    socket_.bind(endpoint);
    socket_.set_option(
        boost::asio::ip::multicast::join_group(asio::ip::make_address(multicast_address)));
  }

  udp::endpoint local_endpoint() const { return socket_.local_endpoint(); }

  void add_session(source_identifier key) {
    sessions.insert(std::make_pair(key, std::make_shared<session>()));
  }

  void start(on_receive_func on_receive) {
    this->on_receive = on_receive;

    running = true;

    start_receive();

    io_thread.reset(new std::thread([&]() { io_context.run(); }));

    handling_thread.reset(new std::thread([&]() {
      while (running.load()) {
        handle_receive();
      }
    }));
  }

  void stop() {
    io_context.stop();
    if (io_thread && io_thread->joinable()) {
      io_thread->join();
    }
    {
      std::lock_guard<std::mutex> lock(mtx);
      running = false;
    }
    cv.notify_one();
    if (handling_thread && handling_thread->joinable()) {
      handling_thread->join();
    }
  }

  void start_receive() {
    packet_data *packet = packet_pool.obtain_packet();

    socket_.async_receive_from(boost::asio::buffer(packet->recv_buf), remote_endpoint_,
                               [packet, this](boost::system::error_code ec, std::size_t length) {
                                 if (!ec) {
                                   start_receive();
                                   packet->size = length;
                                   {
                                     std::lock_guard<std::mutex> lock(mtx);
                                     packet_queue.push(packet);
                                   }
                                   cv.notify_one();
                                 }
                               });
  }

  void handle_receive() {
    packet_data *packet;

    {
      std::unique_lock<std::mutex> lock(mtx);

      cv.wait(lock, [this] { return !packet_queue.empty() || !running; });

      if (!running) {
        return;
      }

      packet = packet_queue.front();
      packet_queue.pop();
    }

    size_t receive_size = packet->size;

    std::stringstream ss(std::string(packet->recv_buf.data(), receive_size));
    ss.read((char *)&packet->flags, sizeof(packet->flags));
    ss.read((char *)&packet->counter, sizeof(packet->counter));
    ss.read((char *)&packet->timestamp, sizeof(packet->timestamp));
    ss.read((char *)&packet->id.stream_unique_id, sizeof(packet->id.stream_unique_id));
    ss.read((char *)&packet->id.data_id, sizeof(packet->id.data_id));

    reordering.commit_packet(packet);

    std::vector<packet_data *> completed_packets;

    for (packet_data *packet = nullptr; (packet = reordering.receive_packet()) != nullptr;) {
      auto it = sessions.find(packet->id);
      if (it == sessions.end()) {
        continue;
      }

      const size_t header_size = 24;
      const size_t payload_size = packet->size - header_size;

      auto &session = *it->second;
      session.buffer.commit(boost::asio::buffer_copy(
          session.buffer.prepare(payload_size),
          boost::asio::buffer(packet->recv_buf.data() + header_size, payload_size)));

      bool completed = packet->flags & 0x1;
      if (completed) {
        if (!session.lost_packet) {
          on_receive(packet->timestamp, packet->id, session.buffer);
        }
        session.buffer.consume(session.buffer.size());
        session.lost_packet = false;
      }
      completed_packets.push_back(packet);
    }

    if (reordering.detect_lost_packet()) {
      for (auto &p : sessions) {
        auto session = p.second;
        session->lost_packet = true;
      }

      const auto packets = reordering.reset();
      for (auto packet : packets) {
        completed_packets.push_back(packet);
      }
      reordering.set_next_counter(((uint32_t)packet->counter + 1) % 65536);
      spdlog::warn("Packet lost");
    }

    packet_pool.release_packets(completed_packets);
  }
};

class data_stream_tcp_receiver {
  asio::io_context io_context;

  struct session {
    asio::streambuf buffer;
    bool lost_packet;

    session() : lost_packet(true) {}
  };

  typedef std::function<void(double, source_identifier, asio::streambuf &)> on_receive_func;
  tcp::socket socket_;
  tcp::acceptor acceptor_;
  tcp::endpoint remote_endpoint_;
  asio::streambuf receive_buff_;
  reordering_packet_buffer reordering;
  packet_buffer_pool packet_pool;
  std::map<source_identifier, std::shared_ptr<session>> sessions;
  on_receive_func on_receive;
  std::atomic_bool running;
  std::shared_ptr<std::thread> handling_thread;
  std::shared_ptr<std::thread> io_thread;
  std::queue<packet_data *> packet_queue;
  std::mutex mtx;
  std::condition_variable cv;

 public:
  data_stream_tcp_receiver(tcp::endpoint endpoint, bool enable_broadcast = false)
      : io_context(), socket_(io_context), acceptor_(io_context, endpoint), running(false), mtx() {}

  tcp::endpoint local_endpoint() const { return acceptor_.local_endpoint(); }

  void add_session(source_identifier key) {
    sessions.insert(std::make_pair(key, std::make_shared<session>()));
  }

  void start(on_receive_func on_receive) {
    this->on_receive = on_receive;

    running = true;

    start_accept();

    io_thread.reset(new std::thread([&]() { io_context.run(); }));

    handling_thread.reset(new std::thread([&]() {
      while (running.load()) {
        handle_receive();
      }
    }));
  }

  void stop() {
    io_context.stop();
    if (io_thread && io_thread->joinable()) {
      io_thread->join();
    }
    {
      std::lock_guard<std::mutex> lock(mtx);
      running = false;
    }
    cv.notify_one();
    if (handling_thread && handling_thread->joinable()) {
      handling_thread->join();
    }
  }

  void start_accept() {
    acceptor_.async_accept(socket_, [this](const boost::system::error_code &ec) {
      if (ec) {
        spdlog::error("Accept failed: {}", ec.message());
        return;
      }

      start_receive();
    });
  }

  void start_receive_data(packet_data *packet) {
    constexpr auto header_size = 26;

    boost::asio::async_read(
        socket_, receive_buff_, asio::transfer_exactly(packet->size - header_size),
        [packet, this](const boost::system::error_code &ec, std::size_t length) {
          if (!ec) {
            const uint8_t *data = static_cast<const uint8_t *>(receive_buff_.data().data());

            std::copy_n(data, length, packet->recv_buf.begin() + header_size);

            receive_buff_.consume(length);

            start_receive();

            {
              std::lock_guard<std::mutex> lock(mtx);
              packet_queue.push(packet);
            }
            cv.notify_one();
          }
        });
  }

  void start_receive() {
    constexpr auto header_size = 26;
    packet_data *packet = packet_pool.obtain_packet();

    boost::asio::async_read(
        socket_, receive_buff_, asio::transfer_exactly(header_size),
        [packet, this](const boost::system::error_code &ec, std::size_t length) {
          if (!ec) {
            const uint8_t *data = static_cast<const uint8_t *>(receive_buff_.data().data());

            const auto payload_size = (uint32_t)data[24] | ((uint32_t)data[25] << 8);

            packet->size = header_size + payload_size;

            std::copy_n(data, length, packet->recv_buf.begin());

            receive_buff_.consume(length);

            start_receive_data(packet);
          }
        });
  }

  void handle_receive() {
    packet_data *packet;

    {
      std::unique_lock<std::mutex> lock(mtx);

      cv.wait(lock, [this] { return !packet_queue.empty() || !running; });

      if (!running) {
        return;
      }

      packet = packet_queue.front();
      packet_queue.pop();
    }

    size_t receive_size = packet->size;

    std::stringstream ss(std::string(packet->recv_buf.data(), receive_size));
    ss.read((char *)&packet->flags, sizeof(packet->flags));
    ss.read((char *)&packet->counter, sizeof(packet->counter));
    ss.read((char *)&packet->timestamp, sizeof(packet->timestamp));
    ss.read((char *)&packet->id.stream_unique_id, sizeof(packet->id.stream_unique_id));
    ss.read((char *)&packet->id.data_id, sizeof(packet->id.data_id));

    reordering.commit_packet(packet);

    std::vector<packet_data *> completed_packets;

    for (packet_data *packet = nullptr; (packet = reordering.receive_packet()) != nullptr;) {
      auto it = sessions.find(packet->id);
      if (it == sessions.end()) {
        continue;
      }

      const size_t header_size = 26;
      const size_t payload_size = packet->size - header_size;

      auto &session = *it->second;
      session.buffer.commit(boost::asio::buffer_copy(
          session.buffer.prepare(payload_size),
          boost::asio::buffer(packet->recv_buf.data() + header_size, payload_size)));

      bool completed = packet->flags & 0x1;
      if (completed) {
        if (!session.lost_packet) {
          on_receive(packet->timestamp, packet->id, session.buffer);
        }
        session.buffer.consume(session.buffer.size());
        session.lost_packet = false;
      }
      completed_packets.push_back(packet);
    }

    if (reordering.detect_lost_packet()) {
      for (auto &p : sessions) {
        auto session = p.second;
        session->lost_packet = true;
      }

      const auto packets = reordering.reset();
      for (auto packet : packets) {
        completed_packets.push_back(packet);
      }
      reordering.set_next_counter(((uint32_t)packet->counter + 1) % 65536);
      spdlog::warn("Packet lost");
    }

    packet_pool.release_packets(completed_packets);
  }
};
}  // namespace coalsack
