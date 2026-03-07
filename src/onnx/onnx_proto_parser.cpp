#include "coalsack/onnx/onnx_proto_parser.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace coalsack {

namespace {

enum class tensor_proto_field : int {
  dims = 1,
  data_type = 2,
  float_data = 4,
  int32_data = 5,
  int64_data = 7,
  name = 8,
  raw_data = 9,
};

enum class attribute_proto_field : int {
  name = 1,
  float_value = 2,
  int_value = 3,
  string_value = 4,
  tensor_value = 5,
  floats = 7,
  ints = 8,
  type = 20,
};

enum class node_proto_field : int {
  input = 1,
  output = 2,
  name = 3,
  op_type = 4,
  attribute = 5,
};

enum class value_info_proto_field : int {
  name = 1,
};

enum class graph_proto_field : int {
  node = 1,
  initializer = 5,
  input = 11,
  output = 12,
};

enum class model_proto_field : int {
  graph = 7,
};

}  // namespace

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
  protobuf_wire_type wire_type = protobuf_wire_type::varint;
  while (r.read_tag(field_number, wire_type)) {
    switch (static_cast<tensor_proto_field>(field_number)) {
      case tensor_proto_field::dims:
        if (wire_type == protobuf_wire_type::length_delimited) {
          auto [ptr, len] = r.read_length_delimited_span();
          decode_packed_int64(ptr, len, out.dims);
        } else if (wire_type == protobuf_wire_type::varint) {
          out.dims.push_back(static_cast<int64_t>(r.read_varint()));
        } else {
          r.skip_field(wire_type);
        }
        break;
      case tensor_proto_field::data_type:
        out.data_type = static_cast<onnx_data_type>(r.read_varint());
        break;
      case tensor_proto_field::float_data:
        if (wire_type == protobuf_wire_type::length_delimited) {
          auto [ptr, len] = r.read_length_delimited_span();
          decode_packed_float(ptr, len, out.float_data);
        } else if (wire_type == protobuf_wire_type::fixed32) {
          out.float_data.push_back(r.read_float32());
        } else {
          r.skip_field(wire_type);
        }
        break;
      case tensor_proto_field::int32_data:
        if (wire_type == protobuf_wire_type::length_delimited) {
          auto [ptr, len] = r.read_length_delimited_span();
          decode_packed_int32(ptr, len, out.int32_data);
        } else if (wire_type == protobuf_wire_type::varint) {
          out.int32_data.push_back(static_cast<int32_t>(r.read_varint()));
        } else {
          r.skip_field(wire_type);
        }
        break;
      case tensor_proto_field::int64_data:
        if (wire_type == protobuf_wire_type::length_delimited) {
          auto [ptr, len] = r.read_length_delimited_span();
          decode_packed_int64(ptr, len, out.int64_data);
        } else if (wire_type == protobuf_wire_type::varint) {
          out.int64_data.push_back(static_cast<int64_t>(r.read_varint()));
        } else {
          r.skip_field(wire_type);
        }
        break;
      case tensor_proto_field::name:
        out.name = r.read_length_delimited();
        break;
      case tensor_proto_field::raw_data:
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
  protobuf_wire_type wire_type = protobuf_wire_type::varint;
  while (r.read_tag(field_number, wire_type)) {
    switch (static_cast<attribute_proto_field>(field_number)) {
      case attribute_proto_field::name:
        out.name = r.read_length_delimited();
        break;
      case attribute_proto_field::float_value:
        out.f = r.read_float32();
        break;
      case attribute_proto_field::int_value:
        out.i = static_cast<int64_t>(r.read_varint());
        break;
      case attribute_proto_field::string_value:
        out.s = r.read_length_delimited();
        break;
      case attribute_proto_field::tensor_value: {
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_tensor_proto tensor;
        parse_onnx_tensor(ptr, len, tensor);
        out.t = std::move(tensor);
        break;
      }
      case attribute_proto_field::floats:
        if (wire_type == protobuf_wire_type::length_delimited) {
          auto [ptr, len] = r.read_length_delimited_span();
          decode_packed_float(ptr, len, out.floats);
        } else if (wire_type == protobuf_wire_type::fixed32) {
          out.floats.push_back(r.read_float32());
        } else {
          r.skip_field(wire_type);
        }
        break;
      case attribute_proto_field::ints:
        if (wire_type == protobuf_wire_type::length_delimited) {
          auto [ptr, len] = r.read_length_delimited_span();
          decode_packed_int64(ptr, len, out.ints);
        } else if (wire_type == protobuf_wire_type::varint) {
          out.ints.push_back(static_cast<int64_t>(r.read_varint()));
        } else {
          r.skip_field(wire_type);
        }
        break;
      case attribute_proto_field::type:
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
  protobuf_wire_type wire_type = protobuf_wire_type::varint;
  while (r.read_tag(field_number, wire_type)) {
    switch (static_cast<node_proto_field>(field_number)) {
      case node_proto_field::input:
        out.input.push_back(r.read_length_delimited());
        break;
      case node_proto_field::output:
        out.output.push_back(r.read_length_delimited());
        break;
      case node_proto_field::name:
        out.name = r.read_length_delimited();
        break;
      case node_proto_field::op_type:
        out.op_type = r.read_length_delimited();
        break;
      case node_proto_field::attribute: {
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
  protobuf_wire_type wire_type = protobuf_wire_type::varint;
  while (r.read_tag(field_number, wire_type)) {
    switch (static_cast<value_info_proto_field>(field_number)) {
      case value_info_proto_field::name:
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
  protobuf_wire_type wire_type = protobuf_wire_type::varint;
  while (r.read_tag(field_number, wire_type)) {
    switch (static_cast<graph_proto_field>(field_number)) {
      case graph_proto_field::node: {
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_node_proto node;
        parse_onnx_node(ptr, len, node);
        out.node.push_back(std::move(node));
        break;
      }
      case graph_proto_field::initializer: {
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_tensor_proto tensor;
        parse_onnx_tensor(ptr, len, tensor);
        out.initializer.push_back(std::move(tensor));
        break;
      }
      case graph_proto_field::input: {
        auto [ptr, len] = r.read_length_delimited_span();
        onnx_value_info_proto vi;
        parse_onnx_value_info(ptr, len, vi);
        out.input.push_back(std::move(vi));
        break;
      }
      case graph_proto_field::output: {
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
  protobuf_wire_type wire_type = protobuf_wire_type::varint;
  while (r.read_tag(field_number, wire_type)) {
    switch (static_cast<model_proto_field>(field_number)) {
      case model_proto_field::graph: {
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
