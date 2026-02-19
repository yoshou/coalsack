#include "coalsack/onnx/onnx_proto_parser.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace coalsack {

// Helper: decode packed repeated int64 values from a length-delimited blob.
static void decode_packed_int64(const uint8_t* data, size_t size, std::vector<int64_t>& out) {
  onnx_proto_reader r(data, size);
  while (!r.eof()) {
    uint64_t v = r.read_varint();
    out.push_back(static_cast<int64_t>(v));
  }
}

// Helper: decode packed repeated int32 values from a length-delimited blob.
static void decode_packed_int32(const uint8_t* data, size_t size, std::vector<int32_t>& out) {
  onnx_proto_reader r(data, size);
  while (!r.eof()) {
    uint64_t v = r.read_varint();
    out.push_back(static_cast<int32_t>(v));
  }
}

// Helper: decode packed repeated float values from a length-delimited blob.
static void decode_packed_float(const uint8_t* data, size_t size, std::vector<float>& out) {
  onnx_proto_reader r(data, size);
  while (!r.eof()) {
    out.push_back(r.read_float32());
  }
}

bool parse_onnx_tensor(const uint8_t* data, size_t size, onnx_tensor_proto& out) {
  onnx_proto_reader r(data, size);
  int field_number = 0;
  int wire_type = 0;
  while (r.read_tag(field_number, wire_type)) {
    switch (field_number) {
      case 1: {  // dims (packed int64)
        auto [ptr, len] = r.read_length_delimited_span();
        decode_packed_int64(ptr, len, out.dims);
        break;
      }
      case 2:  // data_type (varint)
        out.data_type = static_cast<onnx_data_type>(r.read_varint());
        break;
      case 4: {  // float_data (packed float)
        auto [ptr, len] = r.read_length_delimited_span();
        decode_packed_float(ptr, len, out.float_data);
        break;
      }
      case 5: {  // int32_data (packed int32)
        auto [ptr, len] = r.read_length_delimited_span();
        decode_packed_int32(ptr, len, out.int32_data);
        break;
      }
      case 7: {  // int64_data (packed int64)
        auto [ptr, len] = r.read_length_delimited_span();
        decode_packed_int64(ptr, len, out.int64_data);
        break;
      }
      case 8:  // name
        out.name = r.read_length_delimited();
        break;
      case 9:  // raw_data
        out.raw_data = r.read_length_delimited();
        break;
      default:
        r.skip_field(wire_type);
        break;
    }
  }
  return true;
}

bool parse_onnx_attribute(const uint8_t* data, size_t size, onnx_attribute_proto& out) {
  onnx_proto_reader r(data, size);
  int field_number = 0;
  int wire_type = 0;
  while (r.read_tag(field_number, wire_type)) {
    switch (field_number) {
      case 1:  // name
        out.name = r.read_length_delimited();
        break;
      case 2:  // f (float, wire type 5 = 32-bit fixed)
        out.f = r.read_float32();
        break;
      case 3:  // i (int64, wire type 0 = varint)
        out.i = static_cast<int64_t>(r.read_varint());
        break;
      case 4:  // s (bytes/string)
        out.s = r.read_length_delimited();
        break;
      case 5: {  // t (TensorProto message)
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_tensor_proto tensor;
        parse_onnx_tensor(ptr, len, tensor);
        out.t = std::move(tensor);
        break;
      }
      case 7: {  // floats (packed float)
        auto [ptr, len] = r.read_length_delimited_span();
        decode_packed_float(ptr, len, out.floats);
        break;
      }
      case 8: {  // ints (packed int64)
        auto [ptr, len] = r.read_length_delimited_span();
        decode_packed_int64(ptr, len, out.ints);
        break;
      }
      case 20:  // type (varint discriminant)
        out.type = static_cast<int>(r.read_varint());
        break;
      default:
        r.skip_field(wire_type);
        break;
    }
  }
  return true;
}

bool parse_onnx_node(const uint8_t* data, size_t size, onnx_node_proto& out) {
  onnx_proto_reader r(data, size);
  int field_number = 0;
  int wire_type = 0;
  while (r.read_tag(field_number, wire_type)) {
    switch (field_number) {
      case 1:  // input (repeated string)
        out.input.push_back(r.read_length_delimited());
        break;
      case 2:  // output (repeated string)
        out.output.push_back(r.read_length_delimited());
        break;
      case 3:  // name
        out.name = r.read_length_delimited();
        break;
      case 4:  // op_type
        out.op_type = r.read_length_delimited();
        break;
      case 5: {  // attribute (repeated AttributeProto)
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_attribute_proto attr;
        parse_onnx_attribute(ptr, len, attr);
        out.attribute.push_back(std::move(attr));
        break;
      }
      default:
        r.skip_field(wire_type);
        break;
    }
  }
  return true;
}

bool parse_onnx_value_info(const uint8_t* data, size_t size, onnx_value_info_proto& out) {
  onnx_proto_reader r(data, size);
  int field_number = 0;
  int wire_type = 0;
  while (r.read_tag(field_number, wire_type)) {
    switch (field_number) {
      case 1:  // name
        out.name = r.read_length_delimited();
        break;
      default:
        r.skip_field(wire_type);
        break;
    }
  }
  return true;
}

bool parse_onnx_graph(const uint8_t* data, size_t size, onnx_graph_proto& out) {
  onnx_proto_reader r(data, size);
  int field_number = 0;
  int wire_type = 0;
  while (r.read_tag(field_number, wire_type)) {
    switch (field_number) {
      case 1: {  // node (repeated NodeProto)
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_node_proto node;
        parse_onnx_node(ptr, len, node);
        out.node.push_back(std::move(node));
        break;
      }
      case 5: {  // initializer (repeated TensorProto)
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_tensor_proto tensor;
        parse_onnx_tensor(ptr, len, tensor);
        out.initializer.push_back(std::move(tensor));
        break;
      }
      case 11: {  // input (repeated ValueInfoProto)
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_value_info_proto vi;
        parse_onnx_value_info(ptr, len, vi);
        out.input.push_back(std::move(vi));
        break;
      }
      case 12: {  // output (repeated ValueInfoProto)
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_value_info_proto vi;
        parse_onnx_value_info(ptr, len, vi);
        out.output.push_back(std::move(vi));
        break;
      }
      default:
        r.skip_field(wire_type);
        break;
    }
  }
  return true;
}

bool parse_onnx_model(const uint8_t* data, size_t size, onnx_model_proto& out) {
  onnx_proto_reader r(data, size);
  int field_number = 0;
  int wire_type = 0;
  while (r.read_tag(field_number, wire_type)) {
    switch (field_number) {
      case 7: {  // graph (GraphProto)
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_graph_proto graph;
        parse_onnx_graph(ptr, len, graph);
        out.graph = std::move(graph);
        break;
      }
      default:
        r.skip_field(wire_type);
        break;
    }
  }
  return true;
}

bool load_onnx_model(const std::string& filepath, onnx_model_proto& out) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }

  std::streamsize file_size = file.tellg();
  if (file_size <= 0) {
    return false;
  }
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(static_cast<size_t>(file_size));
  if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
    return false;
  }

  try {
    return parse_onnx_model(buffer.data(), buffer.size(), out);
  } catch (const std::exception& e) {
    return false;
  }
}

}  // namespace coalsack
