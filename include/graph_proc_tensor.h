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
        
        template <typename DataType>
        class view_type
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
        };

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

        template <int dim = num_dims - 1, typename FromIter, typename ToIter>
        static void copy(FromIter from, ToIter to, const shape_type &shape, const stride_type &from_stride, const stride_type &to_stride)
        {
            if constexpr (dim < 0)
            {
                *to = (decltype(*to))*from;
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
                stride[i] = stride[i - 1] * shape[i];
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

        elem_type at(index_type index) const
        {
            return get(data.begin(), shape, stride, index);
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

        view_type<const elem_type*> transpose(const std::array<uint32_t, num_dims> &axes) const
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

            view_type<const elem_type*> view;
            view.data = data.data();
            view.stride = new_stride;
            view.shape = new_shape;
            return view;
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

    using tensor_u8_4 = coalsack::frame_message<tensor<uint8_t, 4>>;
    using tensor_f32_4 = coalsack::frame_message<tensor<float, 4>>;
}

#define REGISTER_FRAME_MESSAGE_SUBTYPE(type)            \
    CEREAL_REGISTER_TYPE(coalsack::frame_message<type>) \
    CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::frame_message<type>)

REGISTER_FRAME_MESSAGE_SUBTYPE(coalsack::tensor_u8_4)
REGISTER_FRAME_MESSAGE_SUBTYPE(coalsack::tensor_f32_4)
