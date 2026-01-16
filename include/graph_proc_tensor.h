#pragma once

#include <algorithm>
#include <array>
#include <cereal/types/array.hpp>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "graph_proc_img.h"

namespace coalsack {
template <size_t I, typename T>
struct tuple_n {
  template <typename... Args>
  using type = typename tuple_n<I - 1, T>::template type<T, Args...>;
};

template <typename T>
struct tuple_n<0, T> {
  template <typename... Args>
  using type = std::tuple<Args...>;
};

template <typename T, size_t I>
using tuple_of = typename tuple_n<I, T>::template type<>;

template <int num_dims, int block_num_dims, int dim, typename FromIter, typename ToIter,
          typename Func, typename... Indexes>
static void transform_block(FromIter from, ToIter to, const std::array<uint32_t, num_dims> shape,
                            const std::array<uint32_t, num_dims> from_stride,
                            const std::array<uint32_t, num_dims> to_stride,
                            const std::array<uint32_t, block_num_dims> &block_shape,
                            const std::array<uint32_t, block_num_dims> &block_from_stride,
                            const std::array<uint32_t, block_num_dims> &block_to_stride, Func f,
                            Indexes... indexes);

template <int num_dims, int block_num_dims, int dim, typename... Iter, typename Func,
          typename... Indexes>
static void transform_block(
    std::tuple<Iter...> data, const std::array<uint32_t, num_dims> shape,
    const tuple_of<std::array<uint32_t, num_dims>, sizeof...(Iter)> &stride,
    const std::array<uint32_t, block_num_dims> &block_shape,
    const tuple_of<std::array<uint32_t, block_num_dims>, sizeof...(Iter)> &block_stride, Func f,
    Indexes... indexes);

template <int num_dims, uint32_t concat_dim, int dim = num_dims - 1, typename FromIter,
          typename ToIter>
static void concat(const std::vector<FromIter> &from, ToIter to,
                   const std::vector<std::array<uint32_t, num_dims>> &shape,
                   const std::vector<std::array<uint32_t, num_dims>> &from_stride,
                   const std::array<uint32_t, num_dims> &to_stride);

template <int num_dims, int dim = num_dims - 1, typename FromIter, typename ToIter>
static void copy(FromIter from, ToIter to, const std::array<uint32_t, num_dims> &shape,
                 const std::array<uint32_t, num_dims> &from_stride,
                 const std::array<uint32_t, num_dims> &to_stride);

template <typename ElemType>
class storage {
  using elem_type = ElemType;

  elem_type *data_;
  size_t size_;

 public:
  storage() : data_(nullptr), size_(0) {}

  storage(size_t size) : data_(new elem_type[size]), size_(size) {}

  storage(const storage &other) : storage(other.size_) { std::copy_n(other.data_, size_, data_); }

  storage(storage &&other) : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  ~storage() {
    if (data_) {
      delete[] data_;
    }
    data_ = nullptr;
    size_ = 0;
  }

  storage &operator=(const storage &other) {
    data_ = new elem_type[other.size_];
    size_ = other.size_;
    std::copy_n(other.data_, size_, data_);
    return *this;
  }

  storage &operator=(storage &&other) {
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  elem_type *data() { return data_; }
  const elem_type *data() const { return data_; }
  elem_type *begin() { return data_; }
  const elem_type *begin() const { return data_; }
  elem_type *end() { return data_ + size_; }
  const elem_type *end() const { return data_ + size_; }
  size_t size() const { return size_; }

  template <typename Archive>
  void save(Archive &archive) const {
    std::vector<elem_type> data(begin(), end());
    archive(data);
  }

  template <typename Archive>
  void load(Archive &archive) {
    std::vector<elem_type> data;
    archive(data);

    size_ = data.size();
    if (data_) {
      delete[] data_;
    }
    data_ = new elem_type[size_];
    std::copy_n(data.begin(), size_, data_);
  }
};

template <typename ElemType, int num_dims, typename StorageType = storage<ElemType>>
class tensor {
  static_assert(num_dims >= 1);

 public:
  using elem_type = ElemType;
  using shape_type = std::array<uint32_t, num_dims>;
  using stride_type = std::array<uint32_t, num_dims>;
  using index_type = std::array<uint32_t, num_dims>;
  using this_type = tensor<elem_type, num_dims>;
  using storage_type = StorageType;

  template <typename DataType>
  class view_type_base {
   public:
    DataType data;
    shape_type shape;
    stride_type stride;

    size_t get_size() const { return calculate_size(shape); }
    const elem_type *get_data() const { return data; }
    elem_type *get_data() { return data; }

    tensor contiguous() const {
      tensor new_tensor(shape);
      coalsack::copy<num_dims>(data, new_tensor.data.begin(), shape, stride, new_tensor.stride);
      return new_tensor;
    }

    elem_type get(index_type index) const { return tensor::get(data, shape, stride, index); }
    void set(index_type index, const elem_type &value) { set(data, value, shape, stride, index); }

    template <typename Func, typename T>
    void assign(const view_type_base<T> &other, Func f) {
      assert(other.shape == shape);
      this_type::assign(other.data, data, shape, other.stride, stride, f);
    }

    template <typename T>
    void assign(const view_type_base<T> &other) {
      assert(other.shape == shape);
      coalsack::copy<num_dims>(other.data, data, shape, other.stride, stride);
    }

    template <typename Func>
    void assign(Func f) {
      this_type::assign(data, shape, stride, f);
    }

    std::tuple<tensor, tensor<uint64_t, num_dims>> topk(uint32_t k, size_t axis = 0) const {
      assert(axis < num_dims);
      static_assert(num_dims >= 1);
      assert(k <= shape[axis]);

      using value_index_type = std::pair<float, size_t>;

      std::array<uint32_t, num_dims> new_shape = shape;
      new_shape[axis] = k;

      tensor<elem_type, num_dims> values(new_shape);
      tensor<uint64_t, num_dims> indexes(new_shape);

      std::array<uint32_t, 1> block_shape = {shape[axis]};
      std::array<uint32_t, 1> block_stride1 = {stride[axis]};
      std::array<uint32_t, 1> block_stride2 = {values.stride[axis]};
      std::array<uint32_t, 1> block_stride3 = {indexes.stride[axis]};

      std::array<uint32_t, num_dims - 1> access_stride1 = {};
      std::array<uint32_t, num_dims - 1> access_stride2 = {};
      std::array<uint32_t, num_dims - 1> access_stride3 = {};
      std::array<uint32_t, num_dims - 1> access_shape = {};

      size_t j = 0;
      for (size_t i = 0; i < num_dims; i++) {
        if (i != axis) {
          access_stride1[j] = stride[i];
          access_stride2[j] = values.stride[i];
          access_stride3[j] = indexes.stride[i];
          access_shape[j] = shape[i];
          ++j;
        }
      }

      auto &&datas = std::make_tuple(data, values.data.begin(), indexes.data.begin());
      auto &&strides = std::make_tuple(access_stride1, access_stride2, access_stride3);
      auto &&block_strides = std::make_tuple(block_stride1, block_stride2, block_stride3);

      coalsack::transform_block<num_dims - 1, 1, num_dims - 2>(
          datas, access_shape, strides, block_shape, block_strides,
          [k](auto &data, const auto &shape, const auto &stride, auto...) {
            auto &[src_data, values_data, indexes_data] = data;
            const auto &[src_stride, values_stride, indexes_stride] = stride;

            std::vector<value_index_type> value_indexes(shape.at(0));
            for (size_t i = 0; i < value_indexes.size(); i++) {
              value_indexes[i].first = *(src_data + src_stride.at(0) * i);
              value_indexes[i].second = i;
            }
            std::nth_element(value_indexes.begin(), value_indexes.begin() + k, value_indexes.end(),
                             [](const value_index_type &a, const value_index_type &b) {
                               return a.first > b.first;
                             });

            for (size_t i = 0; i < k; i++) {
              *(values_data + values_stride.at(0) * i) = value_indexes.at(i).first;
              *(indexes_data + indexes_stride.at(0) * i) = value_indexes.at(i).second;
            }
          });

      return std::forward_as_tuple(values, indexes);
    }

    template <typename Func>
    tensor transform(Func f) const {
      tensor new_tensor(shape);
      this_type::transform(data, new_tensor.data.begin(), shape, stride, new_tensor.stride, f);
      return new_tensor;
    }

    template <typename Func, typename T>
    tensor transform(const view_type_base<T> &other, Func f) const {
      shape_type new_shape;
      stride_type access_stride1 = stride;
      stride_type access_stride2 = other.stride;
      for (size_t i = 0; i < num_dims; i++) {
        new_shape[i] = std::max(shape[i], other.shape[i]);

        if (new_shape[i] != shape[i]) {
          assert(shape[i] == 1);
          access_stride1[i] = 0;
        }
        if (new_shape[i] != other.shape[i]) {
          assert(other.shape[i] == 1);
          access_stride2[i] = 0;
        }
      }
      tensor new_tensor(new_shape);
      this_type::transform(data, other.data, new_tensor.data.begin(), new_shape, access_stride1,
                           access_stride2, new_tensor.stride, f);
      return new_tensor;
    }

    template <int num_new_dims, typename Func>
    tensor<elem_type, num_dims + num_new_dims> transform_expand(
        const std::array<uint32_t, num_new_dims> &new_dim_shape, Func f) const {
      static_assert(num_new_dims == 1);
      typename tensor<elem_type, num_dims + num_new_dims>::shape_type new_shape;

      for (size_t i = 0; i < num_new_dims; i++) {
        new_shape[i] = new_dim_shape[i];
      }
      for (size_t i = 0; i < num_dims; i++) {
        new_shape[num_new_dims + i] = shape[i];
      }

      tensor<elem_type, num_dims + num_new_dims> new_tensor(new_shape);

      stride_type access_stride1;
      for (size_t i = 0; i < num_dims; i++) {
        access_stride1[i] = new_tensor.stride[num_new_dims + i];
      }

      typename tensor<elem_type, num_new_dims>::stride_type access_stride2;
      access_stride2[0] = 1;
      for (size_t i = 1; i < num_new_dims; i++) {
        access_stride2[i] = access_stride2[i - 1] * new_dim_shape[i - 1];
      }

      typename tensor<elem_type, num_new_dims>::stride_type access_stride3;
      for (size_t i = 0; i < num_new_dims; i++) {
        access_stride3[i] = new_tensor.stride[i];
      }

      tensor::transform_expand<num_new_dims>(data, new_tensor.data.begin(), shape, stride,
                                             access_stride1, new_dim_shape, access_stride2,
                                             access_stride3, f);
      return new_tensor;
    }

    tensor softmax(size_t axis) const {
      tensor new_tensor(shape);

      static_assert(num_dims >= 2);
      assert(axis < num_dims);

      std::array<uint32_t, 1> block_shape = {shape[axis]};
      std::array<uint32_t, 1> block_stride1 = {stride[axis]};
      std::array<uint32_t, 1> block_stride2 = {new_tensor.stride[axis]};

      std::array<uint32_t, num_dims - 1> access_stride1;
      std::array<uint32_t, num_dims - 1> access_stride2;
      std::array<uint32_t, num_dims - 1> access_shape;

      size_t j = 0;
      for (size_t i = 0; i < num_dims; i++) {
        if (i != axis) {
          access_stride1[j] = stride[i];
          access_stride2[j] = new_tensor.stride[i];
          access_shape[j] = shape[i];
          ++j;
        }
      }

      transform_block<num_dims - 1, 1, num_dims - 2>(
          data, new_tensor.data.begin(), access_shape, access_stride1, access_stride2, block_shape,
          block_stride1, block_stride2, [](const auto &src, auto &dst, auto...) {
            elem_type num = 0;
            assert(src.shape.size() == 1);
            assert(dst.shape.size() == 1);
            assert(src.shape[0] == dst.shape[0]);
            for (size_t i = 0; i < src.shape[0]; i++) {
              const auto offset = src.stride[0] * i;
              num += std::exp(src.data[offset]);
            }
            for (size_t i = 0; i < src.shape[0]; i++) {
              const auto src_offset = src.stride[0] * i;
              const auto dst_offset = dst.stride[0] * i;
              dst.data[dst_offset] = std::exp(src.data[src_offset]) / num;
            }
          });

      return new_tensor;
    }

    view_type_base<const elem_type *> transpose(const std::array<uint32_t, num_dims> &axes) const {
      stride_type new_stride = stride;
      for (size_t i = 0; i < axes.size(); i++) {
        new_stride[i] = stride[axes[i]];
      }
      shape_type new_shape = shape;
      for (size_t i = 0; i < axes.size(); i++) {
        new_shape[i] = shape[axes[i]];
      }

      const_view_type view;
      view.data = data;
      view.stride = new_stride;
      view.shape = new_shape;
      return view;
    }

    template <int new_num_dims>
    tensor<elem_type, new_num_dims> reshape(
        const std::array<uint32_t, new_num_dims> &new_shape) const {
      tensor<elem_type, new_num_dims> result(new_shape);

      result.template view<num_dims>(shape).assign(
          *this, [](auto, const auto value, auto...) { return value; });

      return result;
    }

    template <int new_num_dims = num_dims>
    typename tensor<elem_type, new_num_dims>::const_view_type view(const shape_type &shape,
                                                                   const index_type &offset) const {
      typename tensor<elem_type, new_num_dims>::const_view_type view;
      view.data = data;
      for (size_t i = 0; i < num_dims; i++) {
        view.data += offset[i] * stride[i];
      }
      size_t avail_dims = 0;
      for (size_t i = 0; i < num_dims; i++) {
        if (shape[i] > 0) {
          assert(avail_dims < new_num_dims);
          view.stride[avail_dims] = stride[i];
          view.shape[avail_dims] = shape[i];
          avail_dims++;
        }
      }
      assert(avail_dims <= new_num_dims);

      if (avail_dims > 0) {
        for (size_t i = avail_dims; i < new_num_dims; i++) {
          view.stride[i] = view.stride[i - 1];
          view.shape[i] = 1;
        }
      } else {
        for (size_t i = 0; i < new_num_dims; i++) {
          view.stride[i] = 0;
          view.shape[i] = 1;
        }
      }
      return view;
    }

    template <int new_num_dims = num_dims>
    typename tensor<elem_type, new_num_dims>::const_view_type view(const shape_type &shape,
                                                                   const index_type &offset,
                                                                   const shape_type &step) const {
      typename tensor<elem_type, new_num_dims>::const_view_type view;
      view.data = data;
      for (size_t i = 0; i < num_dims; i++) {
        view.data += offset[i] * stride[i];
      }
      size_t avail_dims = 0;
      for (size_t i = 0; i < num_dims; i++) {
        if (shape[i] > 0) {
          assert(avail_dims < new_num_dims);
          view.stride[avail_dims] = stride[i] * step[i];
          view.shape[avail_dims] = shape[i];
          avail_dims++;
        }
      }
      assert(avail_dims <= new_num_dims);

      if (avail_dims > 0) {
        for (size_t i = avail_dims; i < new_num_dims; i++) {
          view.stride[i] = view.stride[i - 1];
          view.shape[i] = 1;
        }
      } else {
        for (size_t i = 0; i < new_num_dims; i++) {
          view.stride[i] = 0;
          view.shape[i] = 1;
        }
      }
      return view;
    }
  };

  using view_type = view_type_base<elem_type *>;
  using const_view_type = view_type_base<const elem_type *>;

  static size_t calculate_size(const shape_type &shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  }

  static stride_type calculate_stride(const shape_type &shape) {
    stride_type stride;
    stride[0] = 1;
    for (size_t i = 1; i < num_dims; i++) {
      stride[i] = stride[i - 1] * shape[i - 1];
    }
    return stride;
  }

  template <int dim = num_dims - 1, typename FromIter>
  static elem_type get(FromIter from, const shape_type &shape, const stride_type &from_stride,
                       const index_type &index) {
    if constexpr (dim < 0) {
      return *from;
    } else {
      const auto from_offset = from_stride.at(dim) * index[dim];
      return get<dim - 1>(from + from_offset, shape, from_stride, index);
    }
  }

  template <int dim = num_dims - 1, typename ToIter>
  static void set(ToIter to, const elem_type &value, const shape_type &shape,
                  const stride_type &to_stride, const index_type &index) {
    if constexpr (dim < 0) {
      *to = value;
    } else {
      const auto to_offset = to_stride.at(dim) * index[dim];
      set<dim - 1>(to + to_offset, value, shape, to_stride, index);
    }
  }

  template <int dim = num_dims - 1, typename FromIter, typename ToIter, typename Func,
            typename... Indexes>
  static void transform(FromIter from, ToIter to, const shape_type &shape,
                        const stride_type &from_stride, const stride_type &to_stride, Func f,
                        Indexes... indexes) {
    if constexpr (dim < 0) {
      *to = f(*from, indexes...);
    } else {
      for (uint32_t i = 0; i < shape.at(dim); i++) {
        const auto from_offset = from_stride.at(dim) * i;
        const auto to_offset = to_stride.at(dim) * i;
        transform<dim - 1>(from + from_offset, to + to_offset, shape, from_stride, to_stride, f, i,
                           indexes...);
      }
    }
  }

  template <int dim = num_dims - 1, typename FromIter1, typename FromIter2, typename ToIter,
            typename Func, typename... Indexes>
  static void transform(FromIter1 from1, FromIter2 from2, ToIter to, const shape_type &shape,
                        const stride_type &from1_stride, const stride_type &from2_stride,
                        const stride_type &to_stride, Func f, Indexes... indexes) {
    if constexpr (dim < 0) {
      *to = f(*from1, *from2, indexes...);
    } else {
      for (uint32_t i = 0; i < shape.at(dim); i++) {
        const auto from1_offset = from1_stride.at(dim) * i;
        const auto from2_offset = from2_stride.at(dim) * i;
        const auto to_offset = to_stride.at(dim) * i;
        transform<dim - 1>(from1 + from1_offset, from2 + from2_offset, to + to_offset, shape,
                           from1_stride, from2_stride, to_stride, f, i, indexes...);
      }
    }
  }

  template <int dim = num_dims - 1, typename ToIter, typename Func, typename... Indexes>
  static void assign(ToIter to, const shape_type &shape, const stride_type &to_stride, Func f,
                     Indexes... indexes) {
    if constexpr (dim < 0) {
      *to = f(*to, indexes...);
    } else {
      for (uint32_t i = 0; i < shape.at(dim); i++) {
        const auto to_offset = to_stride.at(dim) * i;
        assign<dim - 1>(to + to_offset, shape, to_stride, f, i, indexes...);
      }
    }
  }

  template <int dim = num_dims - 1, typename FromIter, typename ToIter, typename Func,
            typename... Indexes>
  static void assign(FromIter from, ToIter to, const shape_type &shape,
                     const stride_type &from_stride, const stride_type &to_stride, Func f,
                     Indexes... indexes) {
    if constexpr (dim < 0) {
      *to = f(*to, *from, indexes...);
    } else {
      for (uint32_t i = 0; i < shape.at(dim); i++) {
        const auto from_offset = from_stride.at(dim) * i;
        const auto to_offset = to_stride.at(dim) * i;
        assign<dim - 1>(from + from_offset, to + to_offset, shape, from_stride, to_stride, f, i,
                        indexes...);
      }
    }
  }

  template <int dim = num_dims - 1, typename FromIter1, typename FromIter2, typename ToIter,
            typename Func, typename... Indexes>
  static void assign(FromIter1 from1, FromIter2 from2, ToIter to, const shape_type &shape,
                     const stride_type &from1_stride, const stride_type &from2_stride,
                     const stride_type &to_stride, Func f, Indexes... indexes) {
    if constexpr (dim < 0) {
      *to = f(*to, *from1, *from2, indexes...);
    } else {
      for (uint32_t i = 0; i < shape.at(dim); i++) {
        const auto from1_offset = from1_stride.at(dim) * i;
        const auto from2_offset = from2_stride.at(dim) * i;
        const auto to_offset = to_stride.at(dim) * i;
        assign<dim - 1>(from1 + from1_offset, from2 + from2_offset, to + to_offset, shape,
                        from1_stride, from2_stride, to_stride, f, i, indexes...);
      }
    }
  }

  template <int new_dims, int dim = num_dims - 1, typename FromIter, typename ToIter, typename Func,
            typename... Indexes>
  static void transform_expand(FromIter from, ToIter to, const shape_type &shape,
                               const stride_type &from_stride, const stride_type &to_stride,
                               const std::array<uint32_t, new_dims> &block_shape,
                               const std::array<uint32_t, new_dims> &block_from_stride,
                               const std::array<uint32_t, new_dims> &block_to_stride, Func f,
                               Indexes... indexes) {
    if constexpr (dim < 0) {
      const auto block = f(*from, indexes...);
      tensor<elem_type, new_dims>::transform(block.begin(), to, block_shape, block_from_stride,
                                             block_to_stride,
                                             [](const auto &value, auto...) { return value; });
    } else {
      for (uint32_t i = 0; i < shape.at(dim); i++) {
        const auto from_offset = from_stride.at(dim) * i;
        const auto to_offset = to_stride.at(dim) * i;
        transform_expand<new_dims, dim - 1>(from + from_offset, to + to_offset, shape, from_stride,
                                            to_stride, block_shape, block_from_stride,
                                            block_to_stride, f, i, indexes...);
      }
    }
  }

  template <uint32_t dim>
  static tensor concat(const std::vector<tensor> &values) {
    assert(values.size() > 0);

    std::vector<const elem_type *> data;
    std::vector<stride_type> stride;
    std::vector<shape_type> shape;

    uint32_t concat_dim_size = 0;
    for (const auto &value : values) {
      data.push_back(value.data.data());
      stride.push_back(value.stride);
      shape.push_back(value.shape);

      for (uint32_t i = 0; i < num_dims; i++) {
        if (i == dim) {
          concat_dim_size += value.shape[i];
        } else {
          assert(shape[0][i] == value.shape[i]);
        }
      }
    }

    shape_type new_shape = shape[0];
    new_shape[dim] = concat_dim_size;

    tensor new_tensor(new_shape);

    coalsack::concat<num_dims, dim>(data, new_tensor.data.begin(), shape, stride,
                                    new_tensor.stride);
    return new_tensor;
  }

  template <uint32_t dim = num_dims - 1>
  static tensor<elem_type, num_dims + 1> stack(const std::vector<tensor> &values) {
    assert(values.size() > 0);

    std::vector<const elem_type *> data;
    std::vector<std::array<uint32_t, num_dims + 1>> strides;
    std::vector<std::array<uint32_t, num_dims + 1>> shapes;

    for (const auto &value : values) {
      data.push_back(value.data.data());

      std::array<uint32_t, num_dims + 1> stride;
      std::array<uint32_t, num_dims + 1> shape;
      for (uint32_t i = 0; i < num_dims + 1; i++) {
        if (i <= dim) {
          stride[i] = value.stride[i];
          shape[i] = value.shape[i];
        } else if (i == dim + 1) {
          stride[i] = value.stride[i - 1];
          shape[i] = 1;
        } else {
          stride[i] = value.stride[i - 1];
          shape[i] = value.shape[i - 1];
        }
      }

      strides.push_back(stride);
      shapes.push_back(shape);
    }

    std::array<uint32_t, num_dims + 1> new_shape;
    for (uint32_t i = 0; i < num_dims + 1; i++) {
      if (i <= dim) {
        new_shape[i] = values[0].shape[i];
      } else if (i == dim + 1) {
        new_shape[i] = values.size();
      } else {
        new_shape[i] = values[0].shape[i - 1];
      }
    }

    tensor<elem_type, num_dims + 1> new_tensor(new_shape);

    coalsack::concat<num_dims + 1, dim + 1>(data, new_tensor.data.begin(), shapes, strides,
                                            new_tensor.stride);
    return new_tensor;
  }

  tensor() : data(), shape() {}

  tensor(const shape_type &shape)
      : data(calculate_size(shape)), shape(shape), stride(calculate_stride(shape)) {}

  tensor(const shape_type &shape, const stride_type &stride)
      : data(calculate_size(stride) * shape.back()), shape(shape), stride(stride) {}

  tensor(const shape_type &shape, const elem_type *data) : tensor(shape) {
    std::copy_n(data, calculate_size(shape), this->data.begin());
  }

  tensor(const shape_type &shape, const elem_type *data, const stride_type &data_stride)
      : tensor(shape) {
    coalsack::copy<num_dims>(data, this->data.begin(), shape, data_stride, stride);
  }

  tensor(const shape_type &shape, const stride_type &stride, const elem_type *data)
      : tensor(shape, stride) {
    coalsack::copy<num_dims>(data, this->data.begin(), shape, stride, stride);
  }

  tensor(const shape_type &shape, const stride_type &stride, const elem_type *data,
         const stride_type &data_stride)
      : tensor(shape, stride) {
    coalsack::copy<num_dims>(data, this->data.begin(), shape, data_stride, stride);
  }

  tensor(const tensor &other) : data(other.data), shape(other.shape), stride(other.stride) {}

  tensor(tensor &&other)
      : data(std::move(other.data)),
        shape(std::move(other.shape)),
        stride(std::move(other.stride)) {}

  tensor &operator=(const tensor &other) {
    data = other.data;
    shape = other.shape;
    stride = other.stride;
    return *this;
  }

  tensor &operator=(tensor &&other) {
    data = std::move(other.data);
    shape = std::move(other.shape);
    stride = std::move(other.stride);
    return *this;
  }

  size_t get_size() const { return calculate_size(shape); }

  uint32_t get_size(uint32_t axis) const { return shape.at(axis); }

  const elem_type *get_data() const { return data.data(); }
  elem_type *get_data() { return data.data(); }

  elem_type get(index_type index) const { return get(data.begin(), shape, stride, index); }
  void set(index_type index, const elem_type &value) {
    set(data.begin(), value, shape, stride, index);
  }

  bool empty() const { return get_size() == 0; }

  template <typename ToType>
  tensor<ToType, num_dims> cast() const {
    tensor<ToType, num_dims> new_tensor(shape);
    coalsack::copy<num_dims>(data.begin(), new_tensor.data.begin(), shape, stride,
                             new_tensor.stride);
    return new_tensor;
  }

  template <typename Func>
  tensor transform(Func f) const {
    tensor new_tensor(shape);
    transform(data.begin(), new_tensor.data.begin(), shape, stride, new_tensor.stride, f);
    return new_tensor;
  }

  template <typename Func>
  tensor transform(const tensor &other, Func f) const {
    shape_type new_shape;
    stride_type access_stride1 = stride;
    stride_type access_stride2 = other.stride;
    for (size_t i = 0; i < num_dims; i++) {
      new_shape[i] = std::max(shape[i], other.shape[i]);

      if (new_shape[i] != shape[i]) {
        assert(shape[i] == 1);
        access_stride1[i] = 0;
      }
      if (new_shape[i] != other.shape[i]) {
        assert(other.shape[i] == 1);
        access_stride2[i] = 0;
      }
    }
    tensor new_tensor(new_shape);
    transform(data.begin(), other.data.begin(), new_tensor.data.begin(), new_shape, access_stride1,
              access_stride2, new_tensor.stride, f);
    return new_tensor;
  }

  template <int num_new_dims, typename Func>
  tensor<elem_type, num_dims + num_new_dims> transform_expand(
      const std::array<uint32_t, num_new_dims> &new_dim_shape, Func f) const {
    static_assert(num_new_dims == 1);
    typename tensor<elem_type, num_dims + num_new_dims>::shape_type new_shape;

    for (size_t i = 0; i < num_new_dims; i++) {
      new_shape[i] = new_dim_shape[i];
    }
    for (size_t i = 0; i < num_dims; i++) {
      new_shape[num_new_dims + i] = shape[i];
    }

    tensor<elem_type, num_dims + num_new_dims> new_tensor(new_shape);

    stride_type access_stride1;
    for (size_t i = 0; i < num_dims; i++) {
      access_stride1[i] = new_tensor.stride[num_new_dims + i];
    }

    typename tensor<elem_type, num_new_dims>::stride_type access_stride2;
    access_stride2[0] = 1;
    for (size_t i = 1; i < num_new_dims; i++) {
      access_stride2[i] = access_stride2[i - 1] * new_dim_shape[i - 1];
    }

    typename tensor<elem_type, num_new_dims>::stride_type access_stride3;
    for (size_t i = 0; i < num_new_dims; i++) {
      access_stride3[i] = new_tensor.stride[i];
    }

    transform_expand<num_new_dims>(data.begin(), new_tensor.data.begin(), shape, stride,
                                   access_stride1, new_dim_shape, access_stride2, access_stride3,
                                   f);
    return new_tensor;
  }

  template <int num_reduced_dims>
  tensor<elem_type, num_dims - num_reduced_dims> sum(
      const std::array<uint32_t, num_reduced_dims> &axes) const {
    static_assert(num_reduced_dims >= 1);
    using reduced_tensor = tensor<elem_type, num_dims - num_reduced_dims>;

    std::array<bool, num_dims> drop;
    std::fill(drop.begin(), drop.end(), false);
    for (size_t i = 0; i < axes.size(); i++) {
      drop[axes[i]] = true;
    }

    typename reduced_tensor::shape_type new_tensor_shape;

    for (size_t i = 0, j = 0; i < num_dims; i++) {
      if (drop[i] == false) {
        new_tensor_shape[j] = shape[i];
        ++j;
      }
    }

    auto new_tensor = reduced_tensor::zeros(new_tensor_shape);
    stride_type new_tensor_assign_stride;

    for (size_t i = 0, j = 0; i < num_dims; i++) {
      if (drop[i] == false) {
        new_tensor_assign_stride[i] = new_tensor.stride[j];
        ++j;
      } else {
        new_tensor_assign_stride[i] = 0;
      }
    }

    assign(data.begin(), new_tensor.data.begin(), shape, stride, new_tensor_assign_stride,
           [](const float value1, const float value2, auto...) { return value1 + value2; });
    return new_tensor;
  }

  template <int num_reduced_dims>
  tensor<elem_type, num_dims - num_reduced_dims> max(
      const std::array<uint32_t, num_reduced_dims> &axes) const {
    static_assert(num_reduced_dims >= 1);
    using reduced_tensor = tensor<elem_type, num_dims - num_reduced_dims>;

    std::array<bool, num_dims> drop;
    std::fill(drop.begin(), drop.end(), false);
    for (size_t i = 0; i < axes.size(); i++) {
      drop[axes[i]] = true;
    }

    typename reduced_tensor::shape_type new_tensor_shape;

    for (size_t i = 0, j = 0; i < num_dims; i++) {
      if (drop[i] == false) {
        new_tensor_shape[j] = shape[i];
        ++j;
      }
    }

    reduced_tensor new_tensor(new_tensor_shape);
    stride_type new_tensor_assign_stride;

    // Initialize with -infinity for max operation
    std::fill_n(new_tensor.data.data(), new_tensor.get_size(),
                -std::numeric_limits<elem_type>::infinity());

    for (size_t i = 0, j = 0; i < num_dims; i++) {
      if (drop[i] == false) {
        new_tensor_assign_stride[i] = new_tensor.stride[j];
        ++j;
      } else {
        new_tensor_assign_stride[i] = 0;
      }
    }

    assign(
        data.begin(), new_tensor.data.begin(), shape, stride, new_tensor_assign_stride,
        [](const float value1, const float value2, auto...) { return std::max(value1, value2); });
    return new_tensor;
  }

  tensor max_pool3d(size_t kernel_size, size_t stride, size_t padding, size_t dilation) const {
    tensor new_tensor(shape);

    for (size_t k = 0; k < shape[2]; k++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t i = 0; i < shape[0]; i++) {
          auto start_u = static_cast<int64_t>(i * stride) - static_cast<int64_t>(padding);
          auto start_v = static_cast<int64_t>(j * stride) - static_cast<int64_t>(padding);
          auto start_w = static_cast<int64_t>(k * stride) - static_cast<int64_t>(padding);

          const auto end_u =
              std::min(static_cast<int64_t>(start_u + (kernel_size - 1) * dilation + 1),
                       static_cast<int64_t>(shape[0]));
          const auto end_v =
              std::min(static_cast<int64_t>(start_v + (kernel_size - 1) * dilation + 1),
                       static_cast<int64_t>(shape[1]));
          const auto end_w =
              std::min(static_cast<int64_t>(start_w + (kernel_size - 1) * dilation + 1),
                       static_cast<int64_t>(shape[2]));

          while (start_u < 0) {
            start_u += dilation;
          }
          while (start_v < 0) {
            start_v += dilation;
          }
          while (start_w < 0) {
            start_w += dilation;
          }

          elem_type max_value = -std::numeric_limits<elem_type>::infinity();

          for (size_t w = static_cast<size_t>(start_w); w < static_cast<size_t>(end_w);
               w += dilation) {
            for (size_t v = static_cast<size_t>(start_v); v < static_cast<size_t>(end_v);
                 v += dilation) {
              for (size_t u = static_cast<size_t>(start_u); u < static_cast<size_t>(end_u);
                   u += dilation) {
                const auto index = w * this->stride[2] + v * this->stride[1] + u * this->stride[0];
                const auto value = get_data()[index];

                if ((value > max_value) || std::isnan(value)) {
                  max_value = value;
                }
              }
            }
          }

          elem_type *output = new_tensor.get_data() + k * new_tensor.stride[2] +
                              j * new_tensor.stride[1] + i * new_tensor.stride[0];
          *output = max_value;
        }
      }
    }

    return new_tensor;
  }

  const_view_type transpose(const std::array<uint32_t, num_dims> &axes) const {
    stride_type new_stride = stride;
    for (size_t i = 0; i < axes.size(); i++) {
      new_stride[i] = stride[axes[i]];
    }
    shape_type new_shape = shape;
    for (size_t i = 0; i < axes.size(); i++) {
      new_shape[i] = shape[axes[i]];
    }

    const_view_type view;
    view.data = data.data();
    view.stride = new_stride;
    view.shape = new_shape;
    return view;
  }

  template <int new_num_dims>
  tensor<elem_type, new_num_dims> reshape_move(const std::array<uint32_t, new_num_dims> &shape) {
    tensor<elem_type, new_num_dims> new_tensor;
    new_tensor.data = std::move(data);
    new_tensor.shape = shape;
    new_tensor.stride = tensor<elem_type, new_num_dims>::calculate_stride(new_tensor.shape);

    assert(new_tensor.get_size() == get_size());
    return new_tensor;
  }

  tensor softmax(size_t axis) const { return view().softmax(axis); }

  template <int new_num_dims>
  typename tensor<elem_type, new_num_dims>::view_type view(
      const std::array<uint32_t, new_num_dims> &shape) {
    typename tensor<elem_type, new_num_dims>::view_type view;
    view.data = data.data();
    view.shape = shape;
    view.stride[0] = 1;
    for (size_t i = 1; i < new_num_dims; i++) {
      view.stride[i] = view.stride[i - 1] * shape[i - 1];
    }
    return view;
  }

  template <int new_num_dims>
  typename tensor<elem_type, new_num_dims>::const_view_type view(
      const std::array<uint32_t, new_num_dims> &shape) const {
    typename tensor<elem_type, new_num_dims>::const_view_type view;
    view.data = data.data();
    view.shape = shape;
    view.stride[0] = 1;
    for (size_t i = 1; i < new_num_dims; i++) {
      view.stride[i] = view.stride[i - 1] * shape[i - 1];
    }
    return view;
  }

  view_type view() {
    view_type view;
    view.data = data.data();
    view.shape = shape;
    view.stride = stride;
    return view;
  }

  const_view_type view() const {
    const_view_type view;
    view.data = data.data();
    view.shape = shape;
    view.stride = stride;
    return view;
  }

  template <int new_num_dims = num_dims>
  typename tensor<elem_type, new_num_dims>::const_view_type view(const shape_type &shape,
                                                                 const index_type &offset) const {
    typename tensor<elem_type, new_num_dims>::const_view_type view;
    view.data = data.data();
    for (size_t i = 0; i < num_dims; i++) {
      view.data += offset[i] * stride[i];
    }
    size_t avail_dims = 0;
    for (size_t i = 0; i < num_dims; i++) {
      if (shape[i] > 0) {
        assert(avail_dims < new_num_dims);
        view.stride[avail_dims] = stride[i];
        view.shape[avail_dims] = shape[i];
        avail_dims++;
      }
    }
    assert(avail_dims <= new_num_dims);

    if (avail_dims > 0) {
      for (size_t i = avail_dims; i < new_num_dims; i++) {
        view.stride[i] = view.stride[i - 1];
        view.shape[i] = 1;
      }
    } else {
      for (size_t i = 0; i < new_num_dims; i++) {
        view.stride[i] = 0;
        view.shape[i] = 1;
      }
    }
    return view;
  }

  template <int new_num_dims = num_dims>
  typename tensor<elem_type, new_num_dims>::view_type view(const shape_type &shape,
                                                           const index_type &offset) {
    typename tensor<elem_type, new_num_dims>::view_type view;
    view.data = data.data();
    for (size_t i = 0; i < num_dims; i++) {
      view.data += offset[i] * stride[i];
    }
    size_t avail_dims = 0;
    for (size_t i = 0; i < num_dims; i++) {
      if (shape[i] > 0) {
        assert(avail_dims < new_num_dims);
        view.stride[avail_dims] = stride[i];
        view.shape[avail_dims] = shape[i];
        avail_dims++;
      }
    }
    assert(avail_dims <= new_num_dims);

    if (avail_dims > 0) {
      for (size_t i = avail_dims; i < new_num_dims; i++) {
        view.stride[i] = view.stride[i - 1];
        view.shape[i] = 1;
      }
    } else {
      for (size_t i = 0; i < new_num_dims; i++) {
        view.stride[i] = 0;
        view.shape[i] = 1;
      }
    }
    return view;
  }

  template <int new_num_dims = num_dims - 1>
  typename tensor<elem_type, new_num_dims>::view_type squeeze(size_t axis) {
    assert(axis <= num_dims);
    assert(shape[axis] == 1);

    typename tensor<elem_type, new_num_dims>::view_type view;
    view.data = data.data();

    size_t j = 0;
    for (size_t i = 0; i < num_dims; i++) {
      if (i != axis) {
        view.stride[j] = stride[i];
        view.shape[j] = shape[i];
        j++;
      }
    }

    return view;
  }

  template <int new_num_dims = num_dims - 1>
  typename tensor<elem_type, new_num_dims>::const_view_type squeeze(size_t axis) const {
    assert(axis < num_dims);
    assert(shape[axis] == 1);

    typename tensor<elem_type, new_num_dims>::const_view_type view;
    view.data = data.data();

    size_t j = 0;
    for (size_t i = 0; i < num_dims; i++) {
      if (i != axis) {
        view.stride[j] = stride[i];
        view.shape[j] = shape[i];
        j++;
      }
    }

    return view;
  }

  template <int new_num_dims = num_dims + 1>
  typename tensor<elem_type, new_num_dims>::const_view_type unsqueeze(
      size_t axis = num_dims) const {
    assert(axis <= num_dims);

    typename tensor<elem_type, new_num_dims>::const_view_type view;
    view.data = data.data();

    size_t j = 0;
    for (size_t i = 0; i < new_num_dims; i++) {
      if (i != axis) {
        view.stride[i] = stride[j];
        view.shape[i] = shape[j];
        j++;
      } else {
        view.stride[i] = 0;
        view.shape[i] = 1;
      }
    }

    return view;
  }

  static tensor zeros(const shape_type &shape) {
    tensor new_tensor(shape);
    assign(new_tensor.data.begin(), new_tensor.shape, new_tensor.stride,
           [](const float value, auto...) { return static_cast<elem_type>(0); });
    return new_tensor;
  }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(data, shape, stride);
  }

 public:
  storage_type data;
  shape_type shape;
  stride_type stride;
};

template <int num_dims, int dim, typename FromIter, typename ToIter>
static void copy(FromIter from, ToIter to, const std::array<uint32_t, num_dims> &shape,
                 const std::array<uint32_t, num_dims> &from_stride,
                 const std::array<uint32_t, num_dims> &to_stride) {
  if constexpr (dim < 0) {
    *to = static_cast<std::decay_t<decltype(*to)>>(*from);
  } else {
    for (uint32_t i = 0; i < shape.at(dim); i++) {
      const auto from_offset = from_stride.at(dim) * i;
      const auto to_offset = to_stride.at(dim) * i;
      coalsack::copy<num_dims, dim - 1>(from + from_offset, to + to_offset, shape, from_stride,
                                        to_stride);
    }
  }
}

template <int num_dims, int block_num_dims, int dim, typename FromIter, typename ToIter,
          typename Func, typename... Indexes>
static void transform_block(FromIter from, ToIter to, const std::array<uint32_t, num_dims> shape,
                            const std::array<uint32_t, num_dims> from_stride,
                            const std::array<uint32_t, num_dims> to_stride,
                            const std::array<uint32_t, block_num_dims> &block_shape,
                            const std::array<uint32_t, block_num_dims> &block_from_stride,
                            const std::array<uint32_t, block_num_dims> &block_to_stride, Func f,
                            Indexes... indexes) {
  if constexpr (dim < 0) {
    using block_view_type = typename tensor<float, block_num_dims>::view_type;
    using const_block_view_type = typename tensor<float, block_num_dims>::const_view_type;

    const_block_view_type from_block;
    from_block.data = &*from;
    from_block.shape = block_shape;
    from_block.stride = block_from_stride;
    block_view_type to_block;
    to_block.data = &*to;
    to_block.shape = block_shape;
    to_block.stride = block_to_stride;

    f(from_block, to_block, indexes...);
  } else {
    for (uint32_t i = 0; i < shape.at(dim); i++) {
      const auto from_offset = from_stride.at(dim) * i;
      const auto to_offset = to_stride.at(dim) * i;
      coalsack::transform_block<num_dims, block_num_dims, dim - 1>(
          from + from_offset, to + to_offset, shape, from_stride, to_stride, block_shape,
          block_from_stride, block_to_stride, f, i, indexes...);
    }
  }
}

template <int dim, typename... T1, typename... T2, std::size_t... I>
constexpr auto offset(const std::tuple<T1...> &t1, const std::tuple<T2...> &t2, uint32_t idx,
                      std::index_sequence<I...>) {
  return std::tuple{(std::get<I>(t1) + std::get<I>(t2).at(dim) * idx)...};
}

template <int dim, typename... T1, typename... T2>
constexpr auto offset(const std::tuple<T1...> &t1, const std::tuple<T2...> &t2, uint32_t idx) {
  return offset<dim>(t1, t2, idx, std::make_index_sequence<sizeof...(T1)>{});
}

template <int num_dims, int block_num_dims, int dim, typename... Iter, typename Func,
          typename... Indexes>
static void transform_block(
    std::tuple<Iter...> data, const std::array<uint32_t, num_dims> shape,
    const tuple_of<std::array<uint32_t, num_dims>, sizeof...(Iter)> &stride,
    const std::array<uint32_t, block_num_dims> &block_shape,
    const tuple_of<std::array<uint32_t, block_num_dims>, sizeof...(Iter)> &block_stride, Func f,
    Indexes... indexes) {
  if constexpr (dim < 0) {
    f(data, block_shape, block_stride, indexes...);
  } else {
    for (uint32_t i = 0; i < shape.at(dim); i++) {
      coalsack::transform_block<num_dims, block_num_dims, dim - 1>(
          offset<dim>(data, stride, i), shape, stride, block_shape, block_stride, f, i, indexes...);
    }
  }
}

template <int num_dims, uint32_t concat_dim, int dim, typename FromIter, typename ToIter>
static void concat(const std::vector<FromIter> &from, ToIter to,
                   const std::vector<std::array<uint32_t, num_dims>> &shape,
                   const std::vector<std::array<uint32_t, num_dims>> &from_stride,
                   const std::array<uint32_t, num_dims> &to_stride) {
  if constexpr (dim == concat_dim) {
    for (size_t i = 0, k = 0; i < from.size(); i++) {
      for (uint32_t j = 0; j < shape.at(i).at(dim); j++, k++) {
        const auto from_offset = from_stride.at(i).at(dim) * j;
        const auto to_offset = to_stride.at(dim) * k;

        coalsack::copy<num_dims, dim - 1>(from.at(i) + from_offset, to + to_offset, shape.at(0),
                                          from_stride.at(i), to_stride);
      }
    }
  } else {
    for (uint32_t i = 0; i < shape.at(0).at(dim); i++) {
      std::vector<FromIter> next_from;
      for (size_t j = 0; j < from.size(); j++) {
        const auto from_offset = from_stride.at(j).at(dim) * i;
        next_from.push_back(from.at(j) + from_offset);
      }
      const auto to_offset = to_stride.at(dim) * i;
      coalsack::concat<num_dims, concat_dim, dim - 1>(next_from, to + to_offset, shape, from_stride,
                                                      to_stride);
    }
  }
}

using tensor_u8_4 = tensor<uint8_t, 4>;
using tensor_f32_4 = tensor<float, 4>;
}  // namespace coalsack

#define REGISTER_FRAME_MESSAGE_SUBTYPE(type) \
  COALSACK_REGISTER_MESSAGE(coalsack::frame_message<type>, coalsack::graph_message)

REGISTER_FRAME_MESSAGE_SUBTYPE(coalsack::tensor_u8_4)
REGISTER_FRAME_MESSAGE_SUBTYPE(coalsack::tensor_f32_4)
