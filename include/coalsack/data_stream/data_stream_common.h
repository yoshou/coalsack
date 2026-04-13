/// @file data_stream_common.h
/// @brief Shared data types for the data_stream layer.
/// @ingroup rpc
#pragma once

#include <cstdint>

namespace coalsack {
/// @brief Identifies the origin of a packet within a data stream.
struct source_identifier {
  int64_t stream_unique_id;
  int32_t data_id;
};

static int operator<(const source_identifier &lhs, const source_identifier &rhs) {
  if (lhs.stream_unique_id == rhs.stream_unique_id) {
    return lhs.data_id < rhs.data_id;
  }
  return lhs.stream_unique_id < rhs.stream_unique_id;
}

static const size_t PACKET_PAYLOAD_SIZE = 1472;
}  // namespace coalsack
