/// @file rpc_common.h
/// @brief RPC wire-protocol message structures.
/// @ingroup rpc
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace coalsack {
/// @defgroup rpc RPC System
/// @brief Lightweight binary RPC layer built on Boost.Asio.
/// @{

/// @brief Wire-format request packet sent by @c rpc_client.
struct request_t {
  uint32_t id;
  uint32_t func;
  uint32_t length;
};

/// @brief Wire-format response packet returned by @c rpc_server.
struct response_t {
  uint32_t id;
  uint32_t code;
  uint32_t length;
};

using rpc_func =
    std::function<uint32_t(uint32_t, const std::vector<uint8_t> &, std::vector<uint8_t> &)>;
}  // namespace coalsack
