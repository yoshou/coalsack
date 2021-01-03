#pragma once

#include <cstdint>
#include <vector>
#include <functional>

struct request_t
{
    uint32_t id;
    uint32_t func;
    uint32_t length;
};

struct response_t
{
    uint32_t id;
    uint32_t code;
    uint32_t length;
};

using rpc_func = std::function<uint32_t(uint32_t, const std::vector<uint8_t> &, std::vector<uint8_t> &)>;