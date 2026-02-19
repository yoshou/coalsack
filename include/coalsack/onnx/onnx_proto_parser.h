#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace coalsack {

// ONNX TensorProto.DataType enum values
enum class onnx_data_type : int32_t {
  undefined = 0,
  float32 = 1,
  uint8 = 2,
  int8 = 3,
  uint16 = 4,
  int16 = 5,
  int32 = 6,
  int64 = 7,
  string_ = 8,
  bool_ = 9,
  float16 = 10,
  float64 = 11,
  uint32 = 12,
  uint64 = 13,
};

struct onnx_tensor_proto {
  std::vector<int64_t> dims;
  onnx_data_type data_type = onnx_data_type::undefined;
  std::vector<float> float_data;
  std::vector<int32_t> int32_data;
  std::vector<int64_t> int64_data;
  std::string raw_data;
  std::string name;
};

struct onnx_attribute_proto {
  std::string name;
  int type = 0;
  std::optional<float> f;
  std::optional<int64_t> i;
  std::optional<std::string> s;
  std::optional<onnx_tensor_proto> t;
  std::vector<float> floats;
  std::vector<int64_t> ints;

  bool has_f() const { return f.has_value(); }
  bool has_i() const { return i.has_value(); }
  bool has_s() const { return s.has_value(); }
  bool has_t() const { return t.has_value(); }
};

struct onnx_node_proto {
  std::vector<std::string> input;
  std::vector<std::string> output;
  std::string name;
  std::string op_type;
  std::vector<onnx_attribute_proto> attribute;
};

struct onnx_value_info_proto {
  std::string name;
};

struct onnx_graph_proto {
  std::vector<onnx_node_proto> node;
  std::vector<onnx_tensor_proto> initializer;
  std::vector<onnx_value_info_proto> input;
  std::vector<onnx_value_info_proto> output;
};

struct onnx_model_proto {
  std::optional<onnx_graph_proto> graph;
  bool has_graph() const { return graph.has_value(); }
};

// Low-level protobuf binary wire format reader.
// Decodes only the wire types needed for ONNX: varint (0), length-delimited (2), 32-bit (5).
class onnx_proto_reader {
 public:
  onnx_proto_reader(const uint8_t* data, size_t size) : data_(data), size_(size), pos_(0) {}

  bool eof() const { return pos_ >= size_; }
  size_t pos() const { return pos_; }

  // Read a varint. Returns 0 and does not advance on underflow.
  uint64_t read_varint() {
    uint64_t result = 0;
    int shift = 0;
    while (pos_ < size_) {
      uint8_t b = data_[pos_++];
      result |= static_cast<uint64_t>(b & 0x7F) << shift;
      if (!(b & 0x80)) {
        return result;
      }
      shift += 7;
      if (shift >= 64) {
        throw std::runtime_error("varint overflow");
      }
    }
    throw std::runtime_error("unexpected end of buffer in varint");
  }

  // Read a 4-byte little-endian float.
  float read_float32() {
    if (pos_ + 4 > size_) {
      throw std::runtime_error("unexpected end of buffer reading float32");
    }
    float v;
    uint8_t buf[4] = {data_[pos_], data_[pos_ + 1], data_[pos_ + 2], data_[pos_ + 3]};
    std::memcpy(&v, buf, 4);
    pos_ += 4;
    return v;
  }

  // Decode a length-delimited field and return the bytes as a string.
  std::string read_length_delimited() {
    uint64_t len = read_varint();
    if (pos_ + len > size_) {
      throw std::runtime_error("unexpected end of buffer in length-delimited field");
    }
    std::string result(reinterpret_cast<const char*>(data_ + pos_), len);
    pos_ += len;
    return result;
  }

  // Return a raw pointer + length for the next length-delimited field without copying.
  std::pair<const uint8_t*, size_t> read_length_delimited_span() {
    uint64_t len = read_varint();
    if (pos_ + len > size_) {
      throw std::runtime_error("unexpected end of buffer in length-delimited field");
    }
    const uint8_t* ptr = data_ + pos_;
    pos_ += len;
    return {ptr, static_cast<size_t>(len)};
  }

  // Decode (field_number, wire_type) from the next tag varint.
  bool read_tag(int& field_number, int& wire_type) {
    if (eof()) return false;
    uint64_t tag = read_varint();
    field_number = static_cast<int>(tag >> 3);
    wire_type = static_cast<int>(tag & 0x7);
    return true;
  }

  // Skip one field based on wire type.
  void skip_field(int wire_type) {
    switch (wire_type) {
      case 0:  // varint
        read_varint();
        break;
      case 1:  // 64-bit
        if (pos_ + 8 > size_) throw std::runtime_error("skip 64-bit underflow");
        pos_ += 8;
        break;
      case 2:  // length-delimited
        read_length_delimited_span();
        break;
      case 5:  // 32-bit
        if (pos_ + 4 > size_) throw std::runtime_error("skip 32-bit underflow");
        pos_ += 4;
        break;
      default:
        throw std::runtime_error("unknown wire type: " + std::to_string(wire_type));
    }
  }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t pos_;
};

// Forward declarations of parse functions
bool parse_onnx_tensor(const uint8_t* data, size_t size, onnx_tensor_proto& out);
bool parse_onnx_attribute(const uint8_t* data, size_t size, onnx_attribute_proto& out);
bool parse_onnx_node(const uint8_t* data, size_t size, onnx_node_proto& out);
bool parse_onnx_value_info(const uint8_t* data, size_t size, onnx_value_info_proto& out);
bool parse_onnx_graph(const uint8_t* data, size_t size, onnx_graph_proto& out);
bool parse_onnx_model(const uint8_t* data, size_t size, onnx_model_proto& out);

// Load an ONNX model from a file.
bool load_onnx_model(const std::string& filepath, onnx_model_proto& out);

}  // namespace coalsack
