#pragma once

#include <memory>
#include <vector>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <iostream>
#include <mutex>

#include "data_stream_common.h"

namespace coalsack
{
    namespace asio = boost::asio;
    using asio::ip::tcp;
    using asio::ip::udp;

    class data_stream_transmitter
    {
        udp::socket socket_;
        udp::endpoint remote_endpoint_;
        uint16_t counter_;
        std::mutex mtx_;
        boost::array<char, PACKET_PAYLOAD_SIZE> buffer_;

    public:
        data_stream_transmitter(asio::io_service &io_service)
            : socket_(io_service), counter_(0)
        {
        }

        void open(std::string address, uint16_t port)
        {
            counter_ = 0;
            socket_.open(udp::v4());
            remote_endpoint_ = udp::endpoint(asio::ip::address_v4::from_string(address), port);
        }
        void open_broadcast(uint16_t port)
        {
            counter_ = 0;
            socket_.open(udp::v4());
            remote_endpoint_ = udp::endpoint(asio::ip::address_v4::broadcast(), port);
            socket_.set_option(asio::ip::udp::socket::reuse_address(true));
            socket_.set_option(asio::socket_base::broadcast(true));
        }
        void close()
        {
            counter_ = 0;
            socket_.close();
            remote_endpoint_ = udp::endpoint(udp::v4(), 0);
        }
        void reset()
        {
            counter_ = 0;
        }
        void send(source_identifier id, double timestamp, const uint8_t *data, size_t length)
        {
            if (remote_endpoint_.port() == 0)
            {
                throw std::logic_error("The socket hasn't opened.");
            }
            std::lock_guard<std::mutex> lock(mtx_);

            uint16_t flags = 0;

            const size_t header_size = 24;
            const size_t max_payload_size = PACKET_PAYLOAD_SIZE - header_size;

            for (size_t pos = 0; pos < length; pos += max_payload_size)
            {
                uint32_t payload_size = std::min(length - pos, (size_t)max_payload_size);
                uint16_t complete = (pos + payload_size) == length;
                flags = (flags & ~(1 << 0)) | (complete << 0);

                uint16_t counter;
                {
                    counter = counter_;
                    counter_ = (uint16_t)(((uint32_t)counter_ + 1) % 65536);
                }

                size_t offset = 0;
                memcpy(&buffer_[offset], &flags, sizeof(flags));
                offset += sizeof(flags);
                memcpy(&buffer_[offset], &counter, sizeof(counter));
                offset += sizeof(counter);
                memcpy(&buffer_[offset], &timestamp, sizeof(timestamp));
                offset += sizeof(timestamp);
                memcpy(&buffer_[offset], &id.stream_unique_id, sizeof(id.stream_unique_id));
                offset += sizeof(id.stream_unique_id);
                memcpy(&buffer_[offset], &id.data_id, sizeof(id.data_id));
                offset += sizeof(id.data_id);
                memcpy(&buffer_[offset], &data[pos], payload_size);

                assert(offset == header_size);

                size_t send_size = socket_.send_to(boost::asio::buffer(buffer_, payload_size + header_size), remote_endpoint_);
                assert(send_size == (payload_size + header_size));
            }
        }
    };
}
