#pragma once

#include <memory>
#include <vector>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <iostream>
#include <mutex>
#include <queue>
#include <spdlog/spdlog.h>

#include "data_stream_common.h"

namespace coalsack
{
    namespace asio = boost::asio;
    using asio::ip::tcp;
    using asio::ip::udp;

    struct packet_data
    {
        boost::array<char, PACKET_PAYLOAD_SIZE> recv_buf;
        double timestamp = 0;
        source_identifier id = {};
        uint16_t counter = 0;
        uint16_t flags = 0;
        uint16_t size = 0;

        packet_data *next;
    };

    using packet_data_ptr = std::shared_ptr<packet_data>;

    class reordering_packet_buffer
    {
        std::vector<std::shared_ptr<packet_data>> allocated_packets;
        packet_data *free_packets;
        packet_data *ordered_packets;
        uint16_t next_counter;
        double timeout;
        double last_packet_timestamp;
        std::mutex free_packets_mtx;

    public:
        reordering_packet_buffer(double timeout = 100.0)
            : free_packets(nullptr), ordered_packets(nullptr), next_counter(0), timeout(timeout), last_packet_timestamp(0), free_packets_mtx()
        {
        }

        void set_next_counter(uint16_t value)
        {
            next_counter = value;
        }

        packet_data *obtain_packet()
        {
            std::lock_guard<std::mutex> lock(free_packets_mtx);

            if (free_packets == nullptr)
            {
                auto new_packet = std::make_shared<packet_data>();
                allocated_packets.push_back(new_packet);
                return new_packet.get();
            }

            auto packet = free_packets;
            free_packets = free_packets->next;
            packet->next = nullptr;
            return packet;
        }

        uint32_t get_comparable_couter(uint16_t counter)
        {
            if (counter < next_counter)
            {
                return counter + 65536;
            }
            return counter;
        }

        void commit_packet(packet_data *packet)
        {
            assert(packet->next == nullptr);

            if (ordered_packets == nullptr || get_comparable_couter(packet->counter) < get_comparable_couter(ordered_packets->counter))
            {
                packet->next = ordered_packets;
                ordered_packets = packet;
                return;
            }

            packet_data *iter_packet = ordered_packets;

            while (iter_packet->next != nullptr)
            {
                if (get_comparable_couter(packet->counter) > get_comparable_couter(iter_packet->counter))
                {
                    break;
                }

                iter_packet = iter_packet->next;
            }

            auto temp_next = iter_packet->next;
            iter_packet->next = packet;
            packet->next = temp_next;
        }

        bool detect_lost_packet()
        {
            if (ordered_packets == nullptr)
            {
                return false;
            }

            packet_data *iter_packet = ordered_packets;

            while (iter_packet != nullptr)
            {
                if ((iter_packet->timestamp - last_packet_timestamp) * 100 / 1000 / 1000 >= timeout)
                {
                    return true;
                }
                iter_packet = iter_packet->next;
            }
            return false;
        }

        packet_data *receive_packet()
        {
            if (ordered_packets == nullptr)
            {
                return nullptr;
            }

            if (ordered_packets->counter == next_counter)
            {
                auto packet = ordered_packets;
                ordered_packets = ordered_packets->next;
                packet->next = nullptr;

                next_counter = (uint16_t)(((uint32_t)next_counter + 1) % 65536);
                last_packet_timestamp = packet->timestamp;

                return packet;
            }

            return nullptr;
        }

        void reset()
        {
            if (ordered_packets == nullptr)
            {
                return;
            }

            while (ordered_packets != nullptr)
            {
                auto packet = ordered_packets;
                ordered_packets = ordered_packets->next;
                packet->next = nullptr;

                next_counter = (uint16_t)(((uint32_t)packet->counter + 1) % 65536);

                free_packet(packet);
            }
        }

        void free_packet(packet_data *packet)
        {
            std::lock_guard<std::mutex> lock(free_packets_mtx);

            assert(packet->next == nullptr);
            packet->next = free_packets;
            free_packets = packet;
        }
    };

    class data_stream_receiver
    {
        asio::io_service io_service;

        struct session
        {
            asio::streambuf buffer;
            bool lost_packet;

            session()
                : lost_packet(true)
            {
            }
        };
        typedef std::function<void(double, source_identifier, asio::streambuf &)> on_receive_func;
        udp::socket socket_;
        udp::endpoint remote_endpoint_;
        reordering_packet_buffer reordering;
        std::map<source_identifier, std::shared_ptr<session>> sessions;
        on_receive_func on_receive;
        std::atomic_bool started;
        std::shared_ptr<std::thread> handling_thread;
        std::shared_ptr<std::thread> io_thread;
        std::queue<packet_data *> packet_queue;
        std::mutex mtx;

    public:
        data_stream_receiver(udp::endpoint endpoint, bool enable_broadcast = false)
            : io_service(), socket_(io_service, endpoint), started(false), mtx()
        {
            if (enable_broadcast)
            {
                socket_.set_option(
                    boost::asio::ip::udp::socket::broadcast(true));
            }
        }
        data_stream_receiver(udp::endpoint endpoint, std::string multicast_address)
            : io_service(), socket_(io_service), started(false), mtx()
        {
            socket_.open(endpoint.protocol());
            socket_.set_option(boost::asio::ip::udp::socket::reuse_address(true));
            socket_.bind(endpoint);
            socket_.set_option(
                boost::asio::ip::multicast::join_group(asio::ip::address_v4::from_string(multicast_address)));
        }

        udp::endpoint local_endpoint() const
        {
            return socket_.local_endpoint();
        }

        void add_session(source_identifier key)
        {
            sessions.insert(std::make_pair(key, std::make_shared<session>()));
        }

        void start(on_receive_func on_receive)
        {
            this->on_receive = on_receive;
            this->started = true;

            start_receive();

            io_thread.reset(new std::thread([&]()
                                            { io_service.run(); }));

            handling_thread.reset(new std::thread([&]()
                                                  {
            while (started.load())
            {
                handle_receive();
            } }));
        }

        void stop()
        {
            io_service.stop();
            if (io_thread && io_thread->joinable())
            {
                io_thread->join();
            }
            this->started = false;
            if (handling_thread && handling_thread->joinable())
            {
                handling_thread->join();
            }
        }

        void start_receive()
        {
            packet_data *packet = reordering.obtain_packet();

            socket_.async_receive_from(
                boost::asio::buffer(packet->recv_buf),
                remote_endpoint_,
                [packet, this](boost::system::error_code ec, std::size_t length)
                {
                    if (!ec)
                    {
                        start_receive();
                        packet->size = length;
                        {
                            std::lock_guard<std::mutex> lock(mtx);
                            packet_queue.push(packet);
                        }
                    }
                });
        }

        void handle_receive()
        {
            packet_data *packet;

            if (packet_queue.empty())
            {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mtx);

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

            while ((packet = reordering.receive_packet()) != nullptr)
            {
                auto it = sessions.find(packet->id);
                if (it == sessions.end())
                {
                    continue;
                }

                const size_t header_size = 24;
                const size_t payload_size = packet->size - header_size;

                auto &session = *it->second;
                session.buffer.commit(boost::asio::buffer_copy(
                    session.buffer.prepare(payload_size),
                    boost::asio::buffer(packet->recv_buf.data() + header_size, payload_size)));

                bool completed = packet->flags & 0x1;
                if (completed)
                {
                    if (!session.lost_packet)
                    {
                        on_receive(packet->timestamp, packet->id, session.buffer);
                    }
                    session.buffer.consume(session.buffer.size());
                    session.lost_packet = false;
                }
                reordering.free_packet(packet);
            }

            if (reordering.detect_lost_packet())
            {
                for (auto &p : sessions)
                {
                    auto session = p.second;
                    session->lost_packet = true;
                }

                reordering.reset();
                spdlog::warn("Packet lost");
            }
        }
    };
}
