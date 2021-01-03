#pragma once

#include <cstdint>

struct source_identifier
{
    int64_t stream_unique_id;
    int32_t data_id;
};

static int operator<(const source_identifier &lhs, const source_identifier &rhs)
{
    if (lhs.stream_unique_id == rhs.stream_unique_id)
    {
        return lhs.data_id < rhs.data_id;
    }
    return lhs.stream_unique_id < rhs.stream_unique_id;
}
