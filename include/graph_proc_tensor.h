#pragma once

#include <vector>
#include <array>
#include <numeric>
#include <cstdint>
#include <string>
#include <algorithm>
#include <cereal/types/array.hpp>

#include "graph_proc.h"
#include "graph_proc_img.h"

namespace coalsack
{

    template <typename ElemType, int num_dims>
    class tensor
    {
        static_assert(num_dims >= 1);

    public:
        using elem_type = ElemType;
        using shape_type = std::array<uint32_t, num_dims>;
        using stride_type = std::array<uint32_t, num_dims>;
        using index_type = std::array<uint32_t, num_dims>;
        using this_type = tensor<elem_type, num_dims>;

        template <typename DataType>
        class view_type_base
        {
        public:
            DataType data;
            shape_type shape;
            stride_type stride;
            
            tensor contiguous() const
            {
                tensor new_tensor(shape);
                copy(data, new_tensor.data.begin(), shape, stride, new_tensor.stride);
                return new_tensor;
            }

            elem_type at(index_type index) const
            {
                return get(data, shape, stride, index);
            }

            template <typename Func, typename T>
            void assign(const view_type_base<T> &other, Func f)
            {
                assert(other.shape == shape);
                this_type::assign(other.data, data, shape, other.stride, stride, f);
            }

            template <typename Func>
            void assign(Func f)
            {
                this_type::assign(data, shape, stride, f);
            }

            std::tuple<tensor, tensor<uint64_t, num_dims>> topk(size_t k) const
            {
                static_assert(num_dims == 1);

                using value_index_type = std::pair<float, size_t>;

                auto data = contiguous();
                std::vector<value_index_type> value_indexes(data.data.size());
                for (size_t i = 0; i < value_indexes.size(); i++)
                {
                    value_indexes[i].first = data.data[i];
                    value_indexes[i].second = i;
                }
                std::nth_element(value_indexes.begin(), value_indexes.begin() + k, value_indexes.end(),
                    [](const value_index_type& a, const value_index_type& b) { return a.first > b.first; });

                tensor values({k});
                this_type::assign(values.data.begin(), values.shape, values.stride, 
                   [&](const float value, const size_t x)
                   {
                       return value_indexes.at(x).first;
                   });

                tensor<uint64_t, num_dims> indexes({k});
                this_type::assign(indexes.data.begin(), indexes.shape, indexes.stride, 
                   [&](const uint64_t value, const size_t x)
                   {
                       return value_indexes.at(x).second;
                   });

                return std::forward_as_tuple(values, indexes);
            }

            template <typename Func>
            tensor transform(Func f) const
            {
                tensor new_tensor(shape);
                this_type::transform(data, new_tensor.data.begin(), shape, stride, new_tensor.stride, f);
                return new_tensor;
            }

            template <int num_new_dims, typename Func>
            tensor<elem_type, num_dims + num_new_dims> transform_expand(const std::array<uint32_t, num_new_dims> &new_dim_shape, Func f) const
            {
                static_assert(num_new_dims == 1);
                typename tensor<elem_type, num_dims + num_new_dims>::shape_type new_shape;

                for (size_t i = 0; i < num_new_dims; i++)
                {
                    new_shape[i] = new_dim_shape[i];
                }
                for (size_t i = 0; i < num_dims; i++)
                {
                    new_shape[num_new_dims + i] = shape[i];
                }

                tensor<elem_type, num_dims + num_new_dims> new_tensor(new_shape);

                stride_type access_stride1;
                for (size_t i = 0; i < num_dims; i++)
                {
                    access_stride1[i] = new_tensor.stride[num_new_dims + i];
                }

                typename tensor<elem_type, num_new_dims>::stride_type access_stride2;
                access_stride2[0] = 1;
                for (size_t i = 1; i < num_new_dims; i++)
                {
                    access_stride2[i] = access_stride2[i - 1] * new_dim_shape[i - 1];
                }

                typename tensor<elem_type, num_new_dims>::stride_type access_stride3;
                for (size_t i = 0; i < num_new_dims; i++)
                {
                    access_stride3[i] = new_tensor.stride[i];
                }

                tensor::transform_expand<num_new_dims>(data, new_tensor.data.begin(), shape, stride, access_stride1, new_dim_shape, access_stride2, access_stride3, f);
                return new_tensor;
            }
        };

        using view_type = view_type_base<elem_type *>;
        using const_view_type = view_type_base<const elem_type *>;

        static size_t calculate_size(const shape_type &shape)
        {
            return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        }

        template <int dim = num_dims - 1, typename FromIter>
        static elem_type get(FromIter from, const shape_type &shape, const stride_type &from_stride, const index_type &index)
        {
            if constexpr (dim < 0)
            {
                return *from;
            }
            else
            {
                const auto from_offset = from_stride.at(dim) * index[dim];
                return get<dim - 1>(from + from_offset, shape, from_stride, index);
            }
        }

        template <int dim = num_dims - 1, typename ToIter>
        static void set(ToIter to, const elem_type &value, const shape_type &shape, const stride_type &to_stride, const index_type &index)
        {
            if constexpr (dim < 0)
            {
                *to = value;
            }
            else
            {
                const auto to_offset = to_stride.at(dim) * index[dim];
                set<dim - 1>(to + to_offset, value, shape, to_stride, index);
            }
        }

        template <int dim = num_dims - 1, typename FromIter, typename ToIter>
        static void copy(FromIter from, ToIter to, const shape_type &shape, const stride_type &from_stride, const stride_type &to_stride)
        {
            if constexpr (dim < 0)
            {
                *to = static_cast<std::decay_t<decltype(*to)>>(*from);
            }
            else
            {
                for (uint32_t i = 0; i < shape.at(dim); i++)
                {
                    const auto from_offset = from_stride.at(dim) * i;
                    const auto to_offset = to_stride.at(dim) * i;
                    copy<dim - 1>(from + from_offset, to + to_offset, shape, from_stride, to_stride);
                }
            }
        }

        template <int dim = num_dims - 1, typename FromIter, typename ToIter, typename Func, typename... Indexes>
        static void transform(FromIter from, ToIter to, const shape_type &shape, const stride_type &from_stride, const stride_type &to_stride, Func f, Indexes... indexes)
        {
            if constexpr (dim < 0)
            {
                *to = f(*from, indexes...);
            }
            else
            {
                for (uint32_t i = 0; i < shape.at(dim); i++)
                {
                    const auto from_offset = from_stride.at(dim) * i;
                    const auto to_offset = to_stride.at(dim) * i;
                    transform<dim - 1>(from + from_offset, to + to_offset, shape, from_stride, to_stride, f, i, indexes...);
                }
            }
        }

        template <int dim = num_dims - 1, typename FromIter1, typename FromIter2, typename ToIter, typename Func, typename... Indexes>
        static void transform(FromIter1 from1, FromIter2 from2, ToIter to, const shape_type &shape, const stride_type &from1_stride, const stride_type &from2_stride, const stride_type &to_stride, Func f, Indexes... indexes)
        {
            if constexpr (dim < 0)
            {
                *to = f(*from1, *from2, indexes...);
            }
            else
            {
                for (uint32_t i = 0; i < shape.at(dim); i++)
                {
                    const auto from1_offset = from1_stride.at(dim) * i;
                    const auto from2_offset = from2_stride.at(dim) * i;
                    const auto to_offset = to_stride.at(dim) * i;
                    transform<dim - 1>(from1 + from1_offset, from2 + from2_offset, to + to_offset, shape, from1_stride, from2_stride, to_stride, f, i, indexes...);
                }
            }
        }

        template <int dim = num_dims - 1, typename ToIter, typename Func, typename... Indexes>
        static void assign(ToIter to, const shape_type &shape, const stride_type &to_stride, Func f, Indexes... indexes)
        {
            if constexpr (dim < 0)
            {
                *to = f(*to, indexes...);
            }
            else
            {
                for (uint32_t i = 0; i < shape.at(dim); i++)
                {
                    const auto to_offset = to_stride.at(dim) * i;
                    assign<dim - 1>(to + to_offset, shape, to_stride, f, i, indexes...);
                }
            }
        }

        template <int dim = num_dims - 1, typename FromIter, typename ToIter, typename Func, typename... Indexes>
        static void assign(FromIter from, ToIter to, const shape_type &shape, const stride_type &from_stride, const stride_type &to_stride, Func f, Indexes... indexes)
        {
            if constexpr (dim < 0)
            {
                *to = f(*to, *from, indexes...);
            }
            else
            {
                for (uint32_t i = 0; i < shape.at(dim); i++)
                {
                    const auto from_offset = from_stride.at(dim) * i;
                    const auto to_offset = to_stride.at(dim) * i;
                    assign<dim - 1>(from + from_offset, to + to_offset, shape, from_stride, to_stride, f, i, indexes...);
                }
            }
        }

        template <int dim = num_dims - 1, typename FromIter1, typename FromIter2, typename ToIter, typename Func, typename... Indexes>
        static void assign(FromIter1 from1, FromIter2 from2, ToIter to, const shape_type &shape, const stride_type &from1_stride, const stride_type &from2_stride, const stride_type &to_stride, Func f, Indexes... indexes)
        {
            if constexpr (dim < 0)
            {
                *to = f(*to, *from1, *from2, indexes...);
            }
            else
            {
                for (uint32_t i = 0; i < shape.at(dim); i++)
                {
                    const auto from1_offset = from1_stride.at(dim) * i;
                    const auto from2_offset = from2_stride.at(dim) * i;
                    const auto to_offset = to_stride.at(dim) * i;
                    assign<dim - 1>(from1 + from1_offset, from2 + from2_offset, to + to_offset, shape, from1_stride, from2_stride, to_stride, f, i, indexes...);
                }
            }
        }

        template <int new_dims, int dim = num_dims - 1, typename FromIter, typename ToIter, typename Func, typename... Indexes>
        static void transform_expand(FromIter from, ToIter to, const shape_type &shape, const stride_type &from_stride, const stride_type &to_stride, const std::array<uint32_t, new_dims> &block_shape, const std::array<uint32_t, new_dims> &block_from_stride, const std::array<uint32_t, new_dims> &block_to_stride, Func f, Indexes... indexes)
        {
            if constexpr (dim < 0)
            {
                const auto block = f(*from, indexes...);
                tensor<elem_type, new_dims>::transform(block.begin(), to, block_shape, block_from_stride, block_to_stride,
                    [](const auto& value, auto...) { return value; });
            }
            else
            {
                for (uint32_t i = 0; i < shape.at(dim); i++)
                {
                    const auto from_offset = from_stride.at(dim) * i;
                    const auto to_offset = to_stride.at(dim) * i;
                    transform_expand<new_dims, dim - 1>(from + from_offset, to + to_offset, shape, from_stride, to_stride, block_shape, block_from_stride, block_to_stride, f, i, indexes...);
                }
            }
        }

        tensor()
            : data(), shape()
        {
        }

        tensor(const shape_type &shape)
            : data(calculate_size(shape)), shape(shape)
        {
            stride[0] = 1;
            for (size_t i = 1; i < num_dims; i++)
            {
                stride[i] = stride[i - 1] * shape[i - 1];
            }
        }

        tensor(const shape_type &shape, const stride_type &stride)
            : data(calculate_size(stride) * shape.back()), shape(shape), stride(stride)
        {
        }

        tensor(const shape_type &shape, const elem_type *data)
            : tensor(shape)
        {
            std::copy_n(data, calculate_size(shape), this->data.begin());
        }

        tensor(const shape_type &shape, const elem_type *data, const stride_type &data_stride)
            : tensor(shape)
        {
            copy(data, this->data.begin(), shape, data_stride, stride);
        }

        tensor(const shape_type &shape, const stride_type &stride, const elem_type *data)
            : tensor(shape, stride)
        {
            copy(data, this->data.begin(), shape, stride, stride);
        }

        tensor(const shape_type &shape, const stride_type &stride, const elem_type *data, const stride_type &data_stride)
            : tensor(shape, stride)
        {
            copy(data, this->data.begin(), shape, data_stride, stride);
        }

        tensor(const tensor &other)
            : data(other.data), shape(other.shape), stride(other.stride)
        {
        }

        tensor(tensor &&other)
            : data(std::move(other.data)), shape(std::move(other.shape)), stride(std::move(other.stride))
        {
        }

        tensor &operator=(const tensor &other)
        {
            data = other.data;
            shape = other.shape;
            stride = other.stride;
            return *this;
        }

        tensor &operator=(tensor &&other)
        {
            data = std::move(other.data);
            shape = std::move(other.shape);
            stride = std::move(other.stride);
            return *this;
        }

        size_t get_size() const
        {
            return calculate_size(shape);
        }

        uint32_t get_size(uint32_t axis) const
        {
            return shape.at(axis);
        }

        const elem_type *get_data() const
        {
            return data.data();
        }
        elem_type *get_data()
        {
            return data.data();
        }

        elem_type get(index_type index) const
        {
            return get(data.begin(), shape, stride, index);
        }
        void set(index_type index, const elem_type& value)
        {
            set(data.begin(), value, shape, stride, index);
        }

        bool empty() const
        {
            return get_size() == 0;
        }

        template <typename ToType>
        tensor<ToType, num_dims> cast() const
        {
            tensor<ToType, num_dims> new_tensor(shape);
            copy(data.begin(), new_tensor.data.begin(), shape, stride, new_tensor.stride);
            return new_tensor;
        }

        template <typename Func>
        tensor transform(Func f) const
        {
            tensor new_tensor(shape);
            transform(data.begin(), new_tensor.data.begin(), shape, stride, new_tensor.stride, f);
            return new_tensor;
        }

        template <typename Func>
        tensor transform(const tensor& other, Func f) const
        {
            shape_type new_shape;
            stride_type access_stride1 = stride;
            stride_type access_stride2 = other.stride;
            for (size_t i = 0; i < num_dims; i++)
            {
                new_shape[i] = std::max(shape[i], other.shape[i]);

                if (new_shape[i] != shape[i])
                {
                    assert(shape[i] == 1);
                    access_stride1[i] = 0;
                }
                if (new_shape[i] != other.shape[i])
                {
                    assert(other.shape[i] == 1);
                    access_stride2[i] = 0;
                }
            }
            tensor new_tensor(new_shape);
            transform(data.begin(), other.data.begin(), new_tensor.data.begin(), new_shape, access_stride1, access_stride2, new_tensor.stride, f);
            return new_tensor;
        }

        template <int num_new_dims, typename Func>
        tensor<elem_type, num_dims + num_new_dims> transform_expand(const std::array<uint32_t, num_new_dims> &new_dim_shape, Func f) const
        {
            static_assert(num_new_dims == 1);
            typename tensor<elem_type, num_dims + num_new_dims>::shape_type new_shape;

            for (size_t i = 0; i < num_new_dims; i++)
            {
                new_shape[i] = new_dim_shape[i];
            }
            for (size_t i = 0; i < num_dims; i++)
            {
                new_shape[num_new_dims + i] = shape[i];
            }

            tensor<elem_type, num_dims + num_new_dims> new_tensor(new_shape);

            stride_type access_stride1;
            for (size_t i = 0; i < num_dims; i++)
            {
                access_stride1[i] = new_tensor.stride[num_new_dims + i];
            }

            typename tensor<elem_type, num_new_dims>::stride_type access_stride2;
            access_stride2[0] = 1;
            for (size_t i = 1; i < num_new_dims; i++)
            {
                access_stride2[i] = access_stride2[i - 1] * new_dim_shape[i - 1];
            }

            typename tensor<elem_type, num_new_dims>::stride_type access_stride3;
            for (size_t i = 0; i < num_new_dims; i++)
            {
                access_stride3[i] = new_tensor.stride[i];
            }

            transform_expand<num_new_dims>(data.begin(), new_tensor.data.begin(), shape, stride, access_stride1, new_dim_shape, access_stride2, access_stride3, f);
            return new_tensor;
        }

        template <int num_reduced_dims>
        tensor<elem_type, num_dims - num_reduced_dims> sum(const std::array<uint32_t, num_reduced_dims> &axes) const
        {
            static_assert(num_reduced_dims >= 1);
            using reduced_tensor = tensor<elem_type, num_dims - num_reduced_dims>;

            std::array<bool, num_dims> drop;
            std::fill(drop.begin(), drop.end(), false);
            for (size_t i = 0; i < axes.size(); i++)
            {
                drop[axes[i]] = true;
            }

            typename reduced_tensor::shape_type new_tensor_shape;
            
            for (size_t i = 0, j = 0; i < num_dims; i++)
            {
                if (drop[i] == false)
                {
                    new_tensor_shape[j] = shape[i];
                    ++j;
                }
            }

            reduced_tensor new_tensor(new_tensor_shape);
            stride_type new_tensor_assign_stride;

            for (size_t i = 0, j = 0; i < num_dims; i++)
            {
                if (drop[i] == false)
                {
                    new_tensor_assign_stride[i] = new_tensor.stride[j];
                    ++j;
                }
                else
                {
                    new_tensor_assign_stride[i] = 0;
                }
            }

            assign(data.begin(), new_tensor.data.begin(), shape, stride, new_tensor_assign_stride,
                   [](const float value1, const float value2, auto...)
                   {
                       return value1 + value2;
                   });
            return new_tensor;
        }

        tensor max_pool3d(size_t kernel_size, size_t stride, size_t padding, size_t dilation) const
        {
            tensor new_tensor(shape);

            for (size_t k = 0; k < shape[2]; k++)
            {
                for (size_t j = 0; j < shape[1]; j++)
                {
                    for (size_t i = 0; i < shape[0]; i++)
                    {
                        int64_t start_u = i * stride - padding;
                        int64_t start_v = j * stride - padding;
                        int64_t start_w = k * stride - padding;

                        int64_t end_u = std::min(start_u + (kernel_size - 1) * dilation + 1, static_cast<size_t>(shape[0]));
                        int64_t end_v = std::min(start_v + (kernel_size - 1) * dilation + 1, static_cast<size_t>(shape[1]));
                        int64_t end_w = std::min(start_w + (kernel_size - 1) * dilation + 1, static_cast<size_t>(shape[2]));

                        while (start_u < 0)
                        {
                            start_u += dilation;
                        }
                        while (start_v < 0)
                        {
                            start_v += dilation;
                        }
                        while (start_w < 0)
                        {
                            start_w += dilation;
                        }

                        elem_type max_value = -std::numeric_limits<elem_type>::infinity();

                        for (size_t w = static_cast<size_t>(start_w); w < static_cast<size_t>(end_w); w += dilation)
                        {
                            for (size_t v = static_cast<size_t>(start_v); v < static_cast<size_t>(end_v); v += dilation)
                            {
                                for (size_t u = static_cast<size_t>(start_u); u < static_cast<size_t>(end_u); u += dilation)
                                {
                                    const auto index = w * this->stride[2] + v * this->stride[1] + u * this->stride[0];
                                    const auto value = get_data()[index];

                                    if ((value > max_value) || std::isnan(value))
                                    {
                                        max_value = value;
                                    }
                                }
                            }
                        }

                        elem_type *output = new_tensor.get_data() + k * new_tensor.stride[2] + j * new_tensor.stride[1] + i * new_tensor.stride[0];
                        *output = max_value;
                    }
                }
            }

            return new_tensor;
        }

        const_view_type transpose(const std::array<uint32_t, num_dims> &axes) const
        {
            stride_type new_stride = stride;
            for (size_t i = 0; i < axes.size(); i++)
            {
                new_stride[i] = stride[axes[i]];
            }
            shape_type new_shape = shape;
            for (size_t i = 0; i < axes.size(); i++)
            {
                new_shape[i] = shape[axes[i]];
            }

            const_view_type view;
            view.data = data.data();
            view.stride = new_stride;
            view.shape = new_shape;
            return view;
        }

        template <int new_num_dims>
        typename tensor<elem_type, new_num_dims>::view_type view(const std::array<uint32_t, new_num_dims> &shape)
        {
            typename tensor<elem_type, new_num_dims>::view_type view;
            view.data = data.data();
            view.shape = shape;
            view.stride[0] = 1;
            for (size_t i = 1; i < new_num_dims; i++)
            {
                view.stride[i] = view.stride[i - 1] * shape[i - 1];
            }
            return view;
        }

        template <int new_num_dims>
        typename tensor<elem_type, new_num_dims>::const_view_type view(const std::array<uint32_t, new_num_dims> &shape) const
        {
            typename tensor<elem_type, new_num_dims>::const_view_type view;
            view.data = data.data();
            view.shape = shape;
            view.stride[0] = 1;
            for (size_t i = 1; i < new_num_dims; i++)
            {
                view.stride[i] = view.stride[i - 1] * shape[i - 1];
            }
            return view;
        }

        view_type view()
        {
            view_type view;
            view.data = data.data();
            view.shape = shape;
            view.stride = stride;
            return view;
        }

        const_view_type view() const
        {
            const_view_type view;
            view.data = data.data();
            view.shape = shape;
            view.stride = stride;
            return view;
        }

        template <int new_num_dims = num_dims>
        typename tensor<elem_type, new_num_dims>::const_view_type view(const shape_type &shape, const index_type &offset) const
        {
            typename tensor<elem_type, new_num_dims>::const_view_type view;
            view.data = data.data();
            for (size_t i = 0; i < num_dims; i++)
            {
                view.data += offset[i] * stride[i];
            }
            size_t avail_dims = 0;
            for (size_t i = 0; i < num_dims; i++)
            {
                if (shape[i] > 0)
                {
                    assert(avail_dims < new_num_dims);
                    view.stride[avail_dims] = stride[i];
                    view.shape[avail_dims] = shape[i];
                    avail_dims++;
                }
            }
            assert(avail_dims <= new_num_dims);

            if (avail_dims > 0)
            {
                for (size_t i = avail_dims; i < new_num_dims; i++)
                {
                    view.stride[i] = view.stride[i - 1];
                    view.shape[i] = 1;
                }
            }
            else
            {
                for (size_t i = 0; i < new_num_dims; i++)
                {
                    view.stride[i] = 0;
                    view.shape[i] = 1;
                }
            }
            return view;
        }

        template <int new_num_dims = num_dims>
        typename tensor<elem_type, new_num_dims>::view_type view(const shape_type &shape, const index_type &offset)
        {
            typename tensor<elem_type, new_num_dims>::view_type view;
            view.data = data.data();
            for (size_t i = 0; i < num_dims; i++)
            {
                view.data += offset[i] * stride[i];
            }
            size_t avail_dims = 0;
            for (size_t i = 0; i < num_dims; i++)
            {
                if (shape[i] > 0)
                {
                    assert(avail_dims < new_num_dims);
                    view.stride[avail_dims] = stride[i];
                    view.shape[avail_dims] = shape[i];
                    avail_dims++;
                }
            }
            assert(avail_dims <= new_num_dims);

            if (avail_dims > 0)
            {
                for (size_t i = avail_dims; i < new_num_dims; i++)
                {
                    view.stride[i] = view.stride[i - 1];
                    view.shape[i] = 1;
                }
            }
            else
            {
                for (size_t i = 0; i < new_num_dims; i++)
                {
                    view.stride[i] = 0;
                    view.shape[i] = 1;
                }
            }
            return view;
        }

        static tensor zeros(const shape_type &shape)
        {
            tensor new_tensor(shape);
            assign(new_tensor.data.begin(), new_tensor.shape, new_tensor.stride,
                      [](const float value, auto...)
                      {
                          return static_cast<elem_type>(0);
                      });
            return new_tensor;
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(data, shape, stride);
        }

    public:
        std::vector<elem_type> data;
        shape_type shape;
        stride_type stride;
    };

    using tensor_u8_4 = tensor<uint8_t, 4>;
    using tensor_f32_4 = tensor<float, 4>;
}

#define REGISTER_FRAME_MESSAGE_SUBTYPE(type)            \
    CEREAL_REGISTER_TYPE(coalsack::frame_message<type>) \
    CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::frame_message<type>)

REGISTER_FRAME_MESSAGE_SUBTYPE(coalsack::tensor_u8_4)
REGISTER_FRAME_MESSAGE_SUBTYPE(coalsack::tensor_f32_4)
