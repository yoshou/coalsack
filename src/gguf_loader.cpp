#include "gguf_loader.h"

#include <cstring>
#include <fstream>
#include <iostream>

namespace coalsack {

// GGUF magic number: "GGUF" (0x46554747)
constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr uint32_t GGUF_VERSION = 3;

// Type name lookup
const char* ggml_type_name(ggml_type type) {
  switch (type) {
    case ggml_type::F32:
      return "F32";
    case ggml_type::F16:
      return "F16";
    case ggml_type::Q4_0:
      return "Q4_0";
    case ggml_type::Q4_1:
      return "Q4_1";
    case ggml_type::Q5_0:
      return "Q5_0";
    case ggml_type::Q5_1:
      return "Q5_1";
    case ggml_type::Q8_0:
      return "Q8_0";
    case ggml_type::Q8_1:
      return "Q8_1";
    case ggml_type::Q2_K:
      return "Q2_K";
    case ggml_type::Q3_K:
      return "Q3_K";
    case ggml_type::Q4_K:
      return "Q4_K";
    case ggml_type::Q5_K:
      return "Q5_K";
    case ggml_type::Q6_K:
      return "Q6_K";
    case ggml_type::Q8_K:
      return "Q8_K";
    default:
      return "UNKNOWN";
  }
}

// Type size (bytes per element or block)
size_t ggml_type_size(ggml_type type) {
  switch (type) {
    case ggml_type::F32:
      return 4;
    case ggml_type::F16:
      return 2;
    case ggml_type::Q4_0:
      return 18;  // block size
    case ggml_type::Q4_1:
      return 20;
    case ggml_type::Q5_0:
      return 22;
    case ggml_type::Q5_1:
      return 24;
    case ggml_type::Q8_0:
      return 34;
    case ggml_type::Q8_1:
      return 36;
    case ggml_type::Q2_K:
      return 256 / 16 + 2 + 2;
    case ggml_type::Q3_K:
      return 256 / 8 + 256 / 4 + 12 + 2;
    case ggml_type::Q4_K:
      return 144;  // Q4_K block
    case ggml_type::Q5_K:
      return 176;
    case ggml_type::Q6_K:
      return 210;
    case ggml_type::Q8_K:
      return 292;
    default:
      return 0;
  }
}

bool ggml_is_quantized(ggml_type type) { return type != ggml_type::F32 && type != ggml_type::F16; }

// Internal implementation
struct gguf_loader::impl {
  std::string file_path;
  bool loaded = false;

  // Header
  uint32_t version = 0;
  uint64_t tensor_count = 0;
  uint64_t kv_count = 0;

  // Metadata storage
  struct kv_entry {
    gguf_type type;
    // Union-like storage
    std::string str_value;
    std::vector<std::string> str_array;
    std::vector<uint32_t> uint32_array;
    std::vector<int32_t> int32_array;
    std::vector<float> float32_array;
    uint64_t uint_value;
    int64_t int_value;
    double float_value;
    bool bool_value;
  };
  std::unordered_map<std::string, kv_entry> metadata;

  // Tensor info storage
  std::unordered_map<std::string, gguf_loader::tensor_info> tensors;

  // File handle for lazy loading
  std::ifstream file;

  bool read_string(std::string& out) {
    uint64_t len;
    if (!file.read(reinterpret_cast<char*>(&len), sizeof(len))) {
      return false;
    }
    if (len > 0) {
      out.resize(len);
      if (!file.read(&out[0], len)) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  bool read_value(T& value) {
    return static_cast<bool>(file.read(reinterpret_cast<char*>(&value), sizeof(T)));
  }

  bool read_kv_entry(kv_entry& entry, gguf_type type) {
    entry.type = type;

    switch (type) {
      case gguf_type::UINT8: {
        uint8_t val;
        if (!read_value(val)) return false;
        entry.uint_value = val;
        return true;
      }
      case gguf_type::INT8: {
        int8_t val;
        if (!read_value(val)) return false;
        entry.int_value = val;
        return true;
      }
      case gguf_type::UINT16: {
        uint16_t val;
        if (!read_value(val)) return false;
        entry.uint_value = val;
        return true;
      }
      case gguf_type::INT16: {
        int16_t val;
        if (!read_value(val)) return false;
        entry.int_value = val;
        return true;
      }
      case gguf_type::UINT32: {
        uint32_t val;
        if (!read_value(val)) return false;
        entry.uint_value = val;
        return true;
      }
      case gguf_type::INT32: {
        int32_t val;
        if (!read_value(val)) return false;
        entry.int_value = val;
        return true;
      }
      case gguf_type::UINT64: {
        uint64_t val;
        if (!read_value(val)) return false;
        entry.uint_value = val;
        return true;
      }
      case gguf_type::INT64: {
        int64_t val;
        if (!read_value(val)) return false;
        entry.int_value = val;
        return true;
      }
      case gguf_type::FLOAT32: {
        float val;
        if (!read_value(val)) return false;
        entry.float_value = val;
        return true;
      }
      case gguf_type::FLOAT64: {
        double val;
        if (!read_value(val)) return false;
        entry.float_value = val;
        return true;
      }
      case gguf_type::BOOL: {
        uint8_t val;
        if (!read_value(val)) return false;
        entry.bool_value = (val != 0);
        return true;
      }
      case gguf_type::STRING: {
        return read_string(entry.str_value);
      }
      case gguf_type::ARRAY: {
        uint32_t array_type_raw;
        if (!read_value(array_type_raw)) return false;
        gguf_type array_type = static_cast<gguf_type>(array_type_raw);

        uint64_t array_len;
        if (!read_value(array_len)) return false;

        if (array_type == gguf_type::STRING) {
          entry.str_array.resize(array_len);
          for (uint64_t i = 0; i < array_len; ++i) {
            if (!read_string(entry.str_array[i])) return false;
          }
        } else if (array_type == gguf_type::UINT32) {
          entry.uint32_array.resize(array_len);
          if (!file.read(reinterpret_cast<char*>(entry.uint32_array.data()),
                         array_len * sizeof(uint32_t))) {
            return false;
          }
        } else if (array_type == gguf_type::INT32) {
          entry.int32_array.resize(array_len);
          if (!file.read(reinterpret_cast<char*>(entry.int32_array.data()),
                         array_len * sizeof(int32_t))) {
            return false;
          }
        } else if (array_type == gguf_type::FLOAT32) {
          entry.float32_array.resize(array_len);
          if (!file.read(reinterpret_cast<char*>(entry.float32_array.data()),
                         array_len * sizeof(float))) {
            return false;
          }
        } else {
          std::cerr << "Warning: Unsupported array type: " << static_cast<uint32_t>(array_type)
                    << "\n";
          return false;
        }
        return true;
      }
      default:
        std::cerr << "Warning: Unsupported metadata type: " << static_cast<uint32_t>(type) << "\n";
        return false;
    }
  }
};

gguf_loader::gguf_loader() : pimpl_(std::make_unique<impl>()) {}

gguf_loader::~gguf_loader() = default;

bool gguf_loader::load(const std::string& path) {
  pimpl_->file_path = path;
  pimpl_->file.open(path, std::ios::binary);

  if (!pimpl_->file) {
    std::cerr << "Error: Failed to open GGUF file: " << path << "\n";
    return false;
  }

  // Read header
  uint32_t magic;
  if (!pimpl_->read_value(magic)) {
    std::cerr << "Error: Failed to read magic number\n";
    return false;
  }

  if (magic != GGUF_MAGIC) {
    std::cerr << "Error: Invalid GGUF magic: 0x" << std::hex << magic << " (expected 0x"
              << GGUF_MAGIC << ")\n"
              << std::dec;
    return false;
  }

  if (!pimpl_->read_value(pimpl_->version)) {
    std::cerr << "Error: Failed to read version\n";
    return false;
  }

  if (pimpl_->version != GGUF_VERSION) {
    std::cerr << "Warning: GGUF version " << pimpl_->version << " (expected " << GGUF_VERSION
              << ")\n";
  }

  if (!pimpl_->read_value(pimpl_->tensor_count)) {
    std::cerr << "Error: Failed to read tensor count\n";
    return false;
  }

  if (!pimpl_->read_value(pimpl_->kv_count)) {
    std::cerr << "Error: Failed to read KV count\n";
    return false;
  }

  std::cout << "GGUF v" << pimpl_->version << ": " << pimpl_->tensor_count << " tensors, "
            << pimpl_->kv_count << " metadata entries\n";

  // Read metadata
  for (uint64_t i = 0; i < pimpl_->kv_count; ++i) {
    std::string key;
    if (!pimpl_->read_string(key)) {
      std::cerr << "Error: Failed to read metadata key " << i << "\n";
      return false;
    }

    uint32_t type_raw;
    if (!pimpl_->read_value(type_raw)) {
      std::cerr << "Error: Failed to read metadata type for key '" << key << "'\n";
      return false;
    }

    impl::kv_entry entry;
    if (!pimpl_->read_kv_entry(entry, static_cast<gguf_type>(type_raw))) {
      std::cerr << "Error: Failed to read metadata value for key '" << key << "'\n";
      return false;
    }

    pimpl_->metadata[key] = std::move(entry);
  }

  // Read tensor info
  for (uint64_t i = 0; i < pimpl_->tensor_count; ++i) {
    tensor_info info;

    if (!pimpl_->read_string(info.name)) {
      std::cerr << "Error: Failed to read tensor name " << i << "\n";
      return false;
    }

    uint32_t n_dims;
    if (!pimpl_->read_value(n_dims)) {
      std::cerr << "Error: Failed to read tensor dims for '" << info.name << "'\n";
      return false;
    }

    info.shape.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      if (!pimpl_->read_value(info.shape[d])) {
        std::cerr << "Error: Failed to read tensor shape[" << d << "] for '" << info.name << "'\n";
        return false;
      }
    }

    uint32_t type_raw;
    if (!pimpl_->read_value(type_raw)) {
      std::cerr << "Error: Failed to read tensor type for '" << info.name << "'\n";
      return false;
    }
    info.type = static_cast<ggml_type>(type_raw);

    if (!pimpl_->read_value(info.offset)) {
      std::cerr << "Error: Failed to read tensor offset for '" << info.name << "'\n";
      return false;
    }

    // Calculate size (approximate, depends on quantization)
    uint64_t n_elements = 1;
    for (auto dim : info.shape) {
      n_elements *= dim;
    }

    size_t type_size = ggml_type_size(info.type);
    if (ggml_is_quantized(info.type)) {
      // Quantized types use block size
      constexpr size_t QK = 32;  // Common block size for K-quants
      info.size = ((n_elements + QK - 1) / QK) * type_size;
    } else {
      info.size = n_elements * type_size;
    }

    pimpl_->tensors[info.name] = info;
  }

  // Calculate alignment offset for tensor data
  size_t alignment = 32;  // GGUF uses 32-byte alignment
  size_t current_pos = pimpl_->file.tellg();
  size_t aligned_pos = (current_pos + alignment - 1) & ~(alignment - 1);

  std::cout << "Tensor data starts at offset: " << aligned_pos << "\n";

  pimpl_->loaded = true;
  return true;
}

bool gguf_loader::is_loaded() const { return pimpl_->loaded; }

uint32_t gguf_loader::get_version() const { return pimpl_->version; }

uint64_t gguf_loader::get_tensor_count() const { return pimpl_->tensor_count; }

uint64_t gguf_loader::get_kv_count() const { return pimpl_->kv_count; }

std::optional<std::string> gguf_loader::get_string(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end() || it->second.type != gguf_type::STRING) {
    return std::nullopt;
  }
  return it->second.str_value;
}

std::optional<uint32_t> gguf_loader::get_uint32(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end()) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(it->second.uint_value);
}

std::optional<uint64_t> gguf_loader::get_uint64(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end()) {
    return std::nullopt;
  }
  return it->second.uint_value;
}

std::optional<int32_t> gguf_loader::get_int32(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end()) {
    return std::nullopt;
  }
  return static_cast<int32_t>(it->second.int_value);
}

std::optional<int64_t> gguf_loader::get_int64(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end()) {
    return std::nullopt;
  }
  return it->second.int_value;
}

std::optional<float> gguf_loader::get_float32(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end()) {
    return std::nullopt;
  }
  return static_cast<float>(it->second.float_value);
}

std::optional<double> gguf_loader::get_float64(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end()) {
    return std::nullopt;
  }
  return it->second.float_value;
}

std::optional<bool> gguf_loader::get_bool(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end() || it->second.type != gguf_type::BOOL) {
    return std::nullopt;
  }
  return it->second.bool_value;
}

std::vector<std::string> gguf_loader::get_array_string(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end() || it->second.type != gguf_type::ARRAY) {
    return {};
  }
  return it->second.str_array;
}

std::vector<uint32_t> gguf_loader::get_array_uint32(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end() || it->second.type != gguf_type::ARRAY) {
    return {};
  }
  return it->second.uint32_array;
}

std::vector<int32_t> gguf_loader::get_array_int32(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end() || it->second.type != gguf_type::ARRAY) {
    return {};
  }
  return it->second.int32_array;
}

std::vector<float> gguf_loader::get_array_float32(const std::string& key) const {
  auto it = pimpl_->metadata.find(key);
  if (it == pimpl_->metadata.end() || it->second.type != gguf_type::ARRAY) {
    return {};
  }
  return it->second.float32_array;
}

std::vector<std::string> gguf_loader::get_metadata_keys() const {
  std::vector<std::string> keys;
  keys.reserve(pimpl_->metadata.size());
  for (const auto& [key, _] : pimpl_->metadata) {
    keys.push_back(key);
  }
  return keys;
}

std::vector<std::string> gguf_loader::get_tensor_names() const {
  std::vector<std::string> names;
  names.reserve(pimpl_->tensors.size());
  for (const auto& [name, _] : pimpl_->tensors) {
    names.push_back(name);
  }
  return names;
}

std::optional<gguf_loader::tensor_info> gguf_loader::get_tensor_info(
    const std::string& name) const {
  auto it = pimpl_->tensors.find(name);
  if (it == pimpl_->tensors.end()) {
    return std::nullopt;
  }
  return it->second;
}

const std::string& gguf_loader::get_file_path() const { return pimpl_->file_path; }

}  // namespace coalsack
