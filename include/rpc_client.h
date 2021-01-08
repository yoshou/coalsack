#pragma once

#include "rpc_common.h"

#include <boost/asio.hpp>
#include <boost/array.hpp>

namespace asio = boost::asio;
using asio::ip::tcp;
using asio::ip::udp;

class rpc_client
{
    tcp::socket socket;

public:
    tcp::endpoint local_endpoint() const
    {
        return socket.local_endpoint();
    }

    rpc_client(asio::io_service &io_service)
        : socket(io_service)
    {
    }

    void connect(std::string ip, unsigned short port)
    {
        boost::system::error_code error;
        socket.connect(tcp::endpoint(asio::ip::address::from_string(ip), port), error);

        if (error)
        {
            throw std::runtime_error("Failed connecting to " + ip + ":" + std::to_string(port) + " (" + error.message() + ")");
        }
    }

    int64_t invoke(uint32_t func, const std::vector<uint8_t> &arg, std::vector<uint8_t> &res)
    {
        boost::system::error_code error;

        request_t request;
        request.func = func;
        request.id = 0;
        request.length = arg.size();
        asio::write(socket, asio::buffer((const char *)&request, sizeof(request_t)), error);
        asio::write(socket, asio::buffer(arg.data(), arg.size()), error);

        asio::streambuf receive_buffer;
        asio::read(socket, receive_buffer, asio::transfer_exactly(sizeof(response_t)), error);

        if (error)
        {
            return -1;
        }

        response_t response = *asio::buffer_cast<const response_t *>(receive_buffer.data());
        receive_buffer.consume(sizeof(response_t));

        if (response.length > 0)
        {
            asio::read(socket, receive_buffer, asio::transfer_exactly(response.length), error);

            const char *data = asio::buffer_cast<const char *>(receive_buffer.data());
            std::copy(data, data + response.length, std::back_inserter(res));
            receive_buffer.consume(response.length);
        }

        return (int64_t)response.code;
    }
};
