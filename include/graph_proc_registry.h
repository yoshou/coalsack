#pragma once

#include <cereal/types/polymorphic.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace coalsack {

class graph_node;
class graph_message;

enum class registered_type_category { node, message };

struct registered_type_info {
  std::string name;
  registered_type_category category;
  std::string base_type;
};

class type_registry {
 public:
  static type_registry& instance() {
    static type_registry reg;
    return reg;
  }

  void register_type(const std::string& name, registered_type_category category,
                     const std::string& base_type = "") {
    registered_type_info info{name, category, base_type};
    types_.push_back(info);
    type_map_[name] = info;
  }

  const std::vector<registered_type_info>& get_registered_types() const { return types_; }

  std::vector<registered_type_info> get_registered_nodes() const {
    std::vector<registered_type_info> nodes;
    for (const auto& info : types_) {
      if (info.category == registered_type_category::node) {
        nodes.push_back(info);
      }
    }
    return nodes;
  }

  std::vector<registered_type_info> get_registered_messages() const {
    std::vector<registered_type_info> messages;
    for (const auto& info : types_) {
      if (info.category == registered_type_category::message) {
        messages.push_back(info);
      }
    }
    return messages;
  }

  bool is_registered(const std::string& name) const {
    return type_map_.find(name) != type_map_.end();
  }

 private:
  type_registry() = default;
  std::vector<registered_type_info> types_;
  std::unordered_map<std::string, registered_type_info> type_map_;
};

namespace detail {

template <typename T>
struct type_registrar {
  type_registrar(const char* name, registered_type_category category, const char* base_type = "") {
    type_registry::instance().register_type(name, category, base_type);
  }
};

}  // namespace detail
}  // namespace coalsack

#define COALSACK_PP_CAT(a, b) COALSACK_PP_CAT_I(a, b)
#define COALSACK_PP_CAT_I(a, b) a##b

#define COALSACK_REGISTER_NODE_TYPE(Type)                                                         \
  CEREAL_REGISTER_TYPE(Type)                                                                      \
  namespace {                                                                                     \
  static ::coalsack::detail::type_registrar<Type> COALSACK_PP_CAT(                                \
      _coalsack_node_registrar_, __COUNTER__)(#Type, ::coalsack::registered_type_category::node); \
  }

#define COALSACK_REGISTER_NODE_TYPE_WITH_NAME(Type, Name)                                        \
  CEREAL_REGISTER_TYPE_WITH_NAME(Type, Name)                                                     \
  namespace {                                                                                    \
  static ::coalsack::detail::type_registrar<Type> COALSACK_PP_CAT(                               \
      _coalsack_node_registrar_, __COUNTER__)(Name, ::coalsack::registered_type_category::node); \
  }

#define COALSACK_REGISTER_NODE_RELATION(Base, Derived) \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Base, Derived)

#define COALSACK_REGISTER_MESSAGE_TYPE(Type)                                                 \
  CEREAL_REGISTER_TYPE(Type)                                                                 \
  namespace {                                                                                \
  static ::coalsack::detail::type_registrar<Type> COALSACK_PP_CAT(                           \
      _coalsack_msg_registrar_, __COUNTER__)(#Type,                                          \
                                             ::coalsack::registered_type_category::message); \
  }

#define COALSACK_REGISTER_MESSAGE_RELATION(Base, Derived) \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Base, Derived)

#define COALSACK_REGISTER_NODE(Type, Base)                                                       \
  CEREAL_REGISTER_TYPE(Type)                                                                     \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Base, Type)                                               \
  namespace {                                                                                    \
  static ::coalsack::detail::type_registrar<Type> COALSACK_PP_CAT(                               \
      _coalsack_node_registrar_, __COUNTER__)(#Type, ::coalsack::registered_type_category::node, \
                                              #Base);                                            \
  }

#define COALSACK_REGISTER_MESSAGE(Type, Base)                                                      \
  CEREAL_REGISTER_TYPE(Type)                                                                       \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Base, Type)                                                 \
  namespace {                                                                                      \
  static ::coalsack::detail::type_registrar<Type> COALSACK_PP_CAT(                                 \
      _coalsack_msg_registrar_, __COUNTER__)(#Type, ::coalsack::registered_type_category::message, \
                                             #Base);                                               \
  }
