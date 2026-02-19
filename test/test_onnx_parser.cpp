#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "coalsack/onnx/onnx_proto_parser.h"

using namespace coalsack;

// Helper: encode a varint into bytes
static void encode_varint(std::vector<uint8_t>& buf, uint64_t value) {
  while (value > 0x7F) {
    buf.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  buf.push_back(static_cast<uint8_t>(value & 0x7F));
}

// Helper: encode a tag (field_number, wire_type)
static void encode_tag(std::vector<uint8_t>& buf, int field_number, int wire_type) {
  encode_varint(buf, (static_cast<uint64_t>(field_number) << 3) | static_cast<uint64_t>(wire_type));
}

// Helper: encode a length-delimited field
static void encode_length_delimited(std::vector<uint8_t>& buf, const std::vector<uint8_t>& data) {
  encode_varint(buf, data.size());
  buf.insert(buf.end(), data.begin(), data.end());
}

static void encode_string_field(std::vector<uint8_t>& buf, int field_number, const std::string& s) {
  encode_tag(buf, field_number, 2);
  encode_varint(buf, s.size());
  buf.insert(buf.end(), s.begin(), s.end());
}

static void encode_varint_field(std::vector<uint8_t>& buf, int field_number, int64_t value) {
  encode_tag(buf, field_number, 0);
  encode_varint(buf, static_cast<uint64_t>(value));
}

static void encode_float32_field(std::vector<uint8_t>& buf, int field_number, float value) {
  encode_tag(buf, field_number, 5);
  uint8_t bytes[4];
  std::memcpy(bytes, &value, 4);
  buf.insert(buf.end(), bytes, bytes + 4);
}

static void encode_message_field(std::vector<uint8_t>& buf, int field_number,
                                 const std::vector<uint8_t>& msg) {
  encode_tag(buf, field_number, 2);
  encode_length_delimited(buf, msg);
}

// ---------------------------------------------------------------------------
// Tests: onnx_proto_reader — varint
// ---------------------------------------------------------------------------

TEST(onnx_proto_reader, varint_single_byte) {
  uint8_t data[] = {0x05};
  onnx_proto_reader r(data, sizeof(data));
  EXPECT_EQ(r.read_varint(), 5u);
  EXPECT_TRUE(r.eof());
}

TEST(onnx_proto_reader, varint_two_bytes) {
  // 300 = 0b100101100 → [0b10101100, 0b00000010] = [0xAC, 0x02]
  uint8_t data[] = {0xAC, 0x02};
  onnx_proto_reader r(data, sizeof(data));
  EXPECT_EQ(r.read_varint(), 300u);
  EXPECT_TRUE(r.eof());
}

TEST(onnx_proto_reader, varint_zero) {
  uint8_t data[] = {0x00};
  onnx_proto_reader r(data, sizeof(data));
  EXPECT_EQ(r.read_varint(), 0u);
  EXPECT_TRUE(r.eof());
}

TEST(onnx_proto_reader, varint_max_7bit) {
  uint8_t data[] = {0x7F};
  onnx_proto_reader r(data, sizeof(data));
  EXPECT_EQ(r.read_varint(), 127u);
}

// ---------------------------------------------------------------------------
// Tests: onnx_proto_reader — tag decoding
// ---------------------------------------------------------------------------

TEST(onnx_proto_reader, read_tag_basic) {
  // field 1, wire type 2 → tag = (1 << 3) | 2 = 10 = 0x0A
  uint8_t data[] = {0x0A};
  onnx_proto_reader r(data, sizeof(data));
  int field_number = 0, wire_type = 0;
  EXPECT_TRUE(r.read_tag(field_number, wire_type));
  EXPECT_EQ(field_number, 1);
  EXPECT_EQ(wire_type, 2);
}

TEST(onnx_proto_reader, read_tag_eof) {
  onnx_proto_reader r(nullptr, 0);
  int field_number = 0, wire_type = 0;
  EXPECT_FALSE(r.read_tag(field_number, wire_type));
}

TEST(onnx_proto_reader, read_tag_large_field_number) {
  // field 20, wire type 0 → tag = (20 << 3) | 0 = 160 = 0xA0, 0x01 (varint)
  uint8_t data[] = {0xA0, 0x01};
  onnx_proto_reader r(data, sizeof(data));
  int field_number = 0, wire_type = 0;
  EXPECT_TRUE(r.read_tag(field_number, wire_type));
  EXPECT_EQ(field_number, 20);
  EXPECT_EQ(wire_type, 0);
}

// ---------------------------------------------------------------------------
// Tests: onnx_proto_reader — skip_field
// ---------------------------------------------------------------------------

TEST(onnx_proto_reader, skip_varint_field) {
  // Two fields: skip varint 42, then read varint 99
  std::vector<uint8_t> buf;
  encode_varint(buf, 42);
  encode_varint(buf, 99);
  onnx_proto_reader r(buf.data(), buf.size());
  r.skip_field(0);  // skip varint
  EXPECT_EQ(r.read_varint(), 99u);
}

TEST(onnx_proto_reader, skip_length_delimited_field) {
  std::vector<uint8_t> buf;
  // length-delimited: 3 bytes "abc"
  encode_varint(buf, 3);
  buf.push_back('a');
  buf.push_back('b');
  buf.push_back('c');
  // then a varint sentinel
  encode_varint(buf, 55);
  onnx_proto_reader r(buf.data(), buf.size());
  r.skip_field(2);
  EXPECT_EQ(r.read_varint(), 55u);
}

TEST(onnx_proto_reader, skip_32bit_field) {
  std::vector<uint8_t> buf = {0x01, 0x02, 0x03, 0x04};
  encode_varint(buf, 77);
  onnx_proto_reader r(buf.data(), buf.size());
  r.skip_field(5);
  EXPECT_EQ(r.read_varint(), 77u);
}

// ---------------------------------------------------------------------------
// Tests: parse_onnx_attribute
// ---------------------------------------------------------------------------

TEST(parse_onnx_attribute, float_attribute) {
  std::vector<uint8_t> buf;
  encode_string_field(buf, 1, "alpha");  // name
  encode_float32_field(buf, 2, 3.14f);   // f
  encode_varint_field(buf, 20, 1);       // type = FLOAT

  onnx_attribute_proto attr;
  EXPECT_TRUE(parse_onnx_attribute(buf.data(), buf.size(), attr));
  EXPECT_EQ(attr.name, "alpha");
  EXPECT_TRUE(attr.has_f());
  EXPECT_NEAR(*attr.f, 3.14f, 1e-5f);
  EXPECT_FALSE(attr.has_i());
}

TEST(parse_onnx_attribute, int_attribute) {
  std::vector<uint8_t> buf;
  encode_string_field(buf, 1, "axis");  // name
  encode_varint_field(buf, 3,
                      -1);  // i = -1 (zigzag not used; signed stored as 64-bit 2's complement)
  encode_varint_field(buf, 20, 2);  // type = INT

  // Note: in protobuf, negative int64 is stored as a 10-byte varint of the 2's complement.
  // For simplicity in this test we store a positive value.
  std::vector<uint8_t> buf2;
  encode_string_field(buf2, 1, "axis");
  encode_varint_field(buf2, 3, 2);   // i = 2
  encode_varint_field(buf2, 20, 2);  // type = INT

  onnx_attribute_proto attr;
  EXPECT_TRUE(parse_onnx_attribute(buf2.data(), buf2.size(), attr));
  EXPECT_EQ(attr.name, "axis");
  EXPECT_TRUE(attr.has_i());
  EXPECT_EQ(*attr.i, 2);
}

TEST(parse_onnx_attribute, string_attribute) {
  std::vector<uint8_t> buf;
  encode_string_field(buf, 1, "mode");      // name
  encode_string_field(buf, 4, "bilinear");  // s
  encode_varint_field(buf, 20, 3);          // type = STRING

  onnx_attribute_proto attr;
  EXPECT_TRUE(parse_onnx_attribute(buf.data(), buf.size(), attr));
  EXPECT_EQ(attr.name, "mode");
  EXPECT_TRUE(attr.has_s());
  EXPECT_EQ(*attr.s, "bilinear");
}

TEST(parse_onnx_attribute, ints_attribute) {
  // ints = [2, 3] packed into a length-delimited blob
  std::vector<uint8_t> packed;
  encode_varint(packed, 2);
  encode_varint(packed, 3);

  std::vector<uint8_t> buf;
  encode_string_field(buf, 1, "strides");
  encode_tag(buf, 8, 2);  // field 8 = ints, wire type 2
  encode_length_delimited(buf, packed);
  encode_varint_field(buf, 20, 7);  // type = INTS

  onnx_attribute_proto attr;
  EXPECT_TRUE(parse_onnx_attribute(buf.data(), buf.size(), attr));
  EXPECT_EQ(attr.name, "strides");
  ASSERT_EQ(attr.ints.size(), 2u);
  EXPECT_EQ(attr.ints[0], 2);
  EXPECT_EQ(attr.ints[1], 3);
}

TEST(parse_onnx_attribute, floats_attribute) {
  // floats = [1.0, 2.0] packed
  std::vector<uint8_t> packed;
  float v1 = 1.0f, v2 = 2.0f;
  uint8_t b1[4], b2[4];
  std::memcpy(b1, &v1, 4);
  std::memcpy(b2, &v2, 4);
  packed.insert(packed.end(), b1, b1 + 4);
  packed.insert(packed.end(), b2, b2 + 4);

  std::vector<uint8_t> buf;
  encode_string_field(buf, 1, "scales");
  encode_tag(buf, 7, 2);  // field 7 = floats
  encode_length_delimited(buf, packed);
  encode_varint_field(buf, 20, 6);  // type = FLOATS

  onnx_attribute_proto attr;
  EXPECT_TRUE(parse_onnx_attribute(buf.data(), buf.size(), attr));
  ASSERT_EQ(attr.floats.size(), 2u);
  EXPECT_NEAR(attr.floats[0], 1.0f, 1e-6f);
  EXPECT_NEAR(attr.floats[1], 2.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Tests: parse_onnx_tensor
// ---------------------------------------------------------------------------

TEST(parse_onnx_tensor, float_tensor_with_raw_data) {
  // dims=[2], data_type=FLOAT(1), raw_data=8 bytes
  float values[2] = {1.5f, 2.5f};
  std::string raw(reinterpret_cast<char*>(values), 8);

  std::vector<uint8_t> packed_dims;
  encode_varint(packed_dims, 2);  // dim = 2

  std::vector<uint8_t> buf;
  encode_tag(buf, 1, 2);  // dims
  encode_length_delimited(buf, packed_dims);
  encode_varint_field(buf, 2, 1);    // data_type = FLOAT
  encode_string_field(buf, 8, "w");  // name
  encode_tag(buf, 9, 2);             // raw_data
  encode_varint(buf, raw.size());
  buf.insert(buf.end(), raw.begin(), raw.end());

  onnx_tensor_proto tensor;
  EXPECT_TRUE(parse_onnx_tensor(buf.data(), buf.size(), tensor));
  ASSERT_EQ(tensor.dims.size(), 1u);
  EXPECT_EQ(tensor.dims[0], 2);
  EXPECT_EQ(tensor.data_type, onnx_data_type::float32);
  EXPECT_EQ(tensor.name, "w");
  ASSERT_EQ(tensor.raw_data.size(), 8u);
  float out[2];
  std::memcpy(out, tensor.raw_data.data(), 8);
  EXPECT_NEAR(out[0], 1.5f, 1e-6f);
  EXPECT_NEAR(out[1], 2.5f, 1e-6f);
}

TEST(parse_onnx_tensor, float_tensor_with_float_data) {
  // float_data packed: [3.0, 4.0]
  float v1 = 3.0f, v2 = 4.0f;
  uint8_t b1[4], b2[4];
  std::memcpy(b1, &v1, 4);
  std::memcpy(b2, &v2, 4);
  std::vector<uint8_t> packed;
  packed.insert(packed.end(), b1, b1 + 4);
  packed.insert(packed.end(), b2, b2 + 4);

  std::vector<uint8_t> packed_dims;
  encode_varint(packed_dims, 2);

  std::vector<uint8_t> buf;
  encode_tag(buf, 1, 2);
  encode_length_delimited(buf, packed_dims);
  encode_varint_field(buf, 2, 1);  // FLOAT
  encode_tag(buf, 4, 2);           // float_data
  encode_length_delimited(buf, packed);

  onnx_tensor_proto tensor;
  EXPECT_TRUE(parse_onnx_tensor(buf.data(), buf.size(), tensor));
  ASSERT_EQ(tensor.float_data.size(), 2u);
  EXPECT_NEAR(tensor.float_data[0], 3.0f, 1e-6f);
  EXPECT_NEAR(tensor.float_data[1], 4.0f, 1e-6f);
}

TEST(parse_onnx_tensor, int64_tensor_with_int64_data) {
  // int64_data packed: [10, 20]
  std::vector<uint8_t> packed;
  encode_varint(packed, 10);
  encode_varint(packed, 20);

  std::vector<uint8_t> packed_dims;
  encode_varint(packed_dims, 2);

  std::vector<uint8_t> buf;
  encode_tag(buf, 1, 2);
  encode_length_delimited(buf, packed_dims);
  encode_varint_field(buf, 2, 7);  // INT64
  encode_tag(buf, 7, 2);           // int64_data
  encode_length_delimited(buf, packed);

  onnx_tensor_proto tensor;
  EXPECT_TRUE(parse_onnx_tensor(buf.data(), buf.size(), tensor));
  EXPECT_EQ(tensor.data_type, onnx_data_type::int64);
  ASSERT_EQ(tensor.int64_data.size(), 2u);
  EXPECT_EQ(tensor.int64_data[0], 10);
  EXPECT_EQ(tensor.int64_data[1], 20);
}

// ---------------------------------------------------------------------------
// Tests: parse_onnx_node
// ---------------------------------------------------------------------------

TEST(parse_onnx_node, basic_node) {
  std::vector<uint8_t> buf;
  encode_string_field(buf, 1, "X");       // input
  encode_string_field(buf, 1, "Y");       // input (2nd)
  encode_string_field(buf, 2, "Z");       // output
  encode_string_field(buf, 3, "mynode");  // name
  encode_string_field(buf, 4, "Add");     // op_type

  onnx_node_proto node;
  EXPECT_TRUE(parse_onnx_node(buf.data(), buf.size(), node));
  ASSERT_EQ(node.input.size(), 2u);
  EXPECT_EQ(node.input[0], "X");
  EXPECT_EQ(node.input[1], "Y");
  ASSERT_EQ(node.output.size(), 1u);
  EXPECT_EQ(node.output[0], "Z");
  EXPECT_EQ(node.name, "mynode");
  EXPECT_EQ(node.op_type, "Add");
}

TEST(parse_onnx_node, node_with_attribute) {
  // Build an attribute sub-message: axis=1
  std::vector<uint8_t> attr_buf;
  encode_string_field(attr_buf, 1, "axis");
  encode_varint_field(attr_buf, 3, 1);   // i = 1
  encode_varint_field(attr_buf, 20, 2);  // type = INT

  std::vector<uint8_t> buf;
  encode_string_field(buf, 1, "X");
  encode_string_field(buf, 2, "Y");
  encode_string_field(buf, 4, "Softmax");
  encode_message_field(buf, 5, attr_buf);  // attribute

  onnx_node_proto node;
  EXPECT_TRUE(parse_onnx_node(buf.data(), buf.size(), node));
  EXPECT_EQ(node.op_type, "Softmax");
  ASSERT_EQ(node.attribute.size(), 1u);
  EXPECT_EQ(node.attribute[0].name, "axis");
  EXPECT_TRUE(node.attribute[0].has_i());
  EXPECT_EQ(*node.attribute[0].i, 1);
}

// ---------------------------------------------------------------------------
// Tests: parse_onnx_model
// ---------------------------------------------------------------------------

TEST(parse_onnx_model, model_with_graph) {
  // Build a minimal graph: one node (Relu), one input "X", one output "Y"
  std::vector<uint8_t> input_vi;
  encode_string_field(input_vi, 1, "X");

  std::vector<uint8_t> output_vi;
  encode_string_field(output_vi, 1, "Y");

  std::vector<uint8_t> node_buf;
  encode_string_field(node_buf, 1, "X");
  encode_string_field(node_buf, 2, "Y");
  encode_string_field(node_buf, 4, "Relu");

  std::vector<uint8_t> graph_buf;
  encode_message_field(graph_buf, 1, node_buf);    // node
  encode_message_field(graph_buf, 11, input_vi);   // input
  encode_message_field(graph_buf, 12, output_vi);  // output

  std::vector<uint8_t> model_buf;
  encode_message_field(model_buf, 7, graph_buf);  // graph

  onnx_model_proto model;
  EXPECT_TRUE(parse_onnx_model(model_buf.data(), model_buf.size(), model));
  EXPECT_TRUE(model.has_graph());
  ASSERT_EQ(model.graph->node.size(), 1u);
  EXPECT_EQ(model.graph->node[0].op_type, "Relu");
  ASSERT_EQ(model.graph->input.size(), 1u);
  EXPECT_EQ(model.graph->input[0].name, "X");
  ASSERT_EQ(model.graph->output.size(), 1u);
  EXPECT_EQ(model.graph->output[0].name, "Y");
}

TEST(parse_onnx_model, empty_model_has_no_graph) {
  std::vector<uint8_t> buf;
  encode_varint_field(buf, 1, 10);  // ir_version only, no graph

  onnx_model_proto model;
  EXPECT_TRUE(parse_onnx_model(buf.data(), buf.size(), model));
  EXPECT_FALSE(model.has_graph());
}

// ---------------------------------------------------------------------------
// Tests: onnx_data_type enum
// ---------------------------------------------------------------------------

TEST(onnx_data_type, enum_values) {
  EXPECT_EQ(static_cast<int32_t>(onnx_data_type::float32), 1);
  EXPECT_EQ(static_cast<int32_t>(onnx_data_type::int32), 6);
  EXPECT_EQ(static_cast<int32_t>(onnx_data_type::int64), 7);
  EXPECT_EQ(static_cast<int32_t>(onnx_data_type::bool_), 9);
  EXPECT_EQ(static_cast<int32_t>(onnx_data_type::float64), 11);
}
