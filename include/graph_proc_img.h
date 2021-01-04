#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "graph_proc.h"

class image
{
public:
    image()
        : data()
        , width(0)
        , height(0)
        , bpp(0)
        , stride(0)
    {}
    
    image(uint32_t width, uint32_t height, uint32_t bpp, uint32_t stride, uint32_t metadata_capacity = 0)
        : data(stride * height + metadata_capacity)
        , width(width)
        , height(height)
        , bpp(bpp)
        , stride(stride)
    {}
    
    image(uint32_t width, uint32_t height, uint32_t bpp, uint32_t stride, const uint8_t* data, uint32_t metadata_capacity = 0)
        : image(width, height, bpp, stride, metadata_capacity)
    {
        std::copy_n(data, stride * height, this->data.begin());
    }

    image(const image& other)
        : data(other.data)
        , width(other.width)
        , height(other.height)
        , bpp(other.bpp)
        , stride(other.stride)
    {}

    image(image&& other)
        : data(std::move(other.data))
        , width(other.width)
        , height(other.height)
        , bpp(other.bpp)
        , stride(other.stride)
    {}

    image& operator=(const image& other)
    {
        data = other.data;
        width = other.width;
        height = other.height;
        bpp = other.bpp;
        stride = other.stride;
        return *this;
    }

    uint32_t get_width() const
    {
        return width;
    }
    uint32_t get_height() const
    {
        return height;
    }
    uint32_t get_bpp() const
    {
        return bpp;
    }
    uint32_t get_stride() const
    {
        return stride;
    }

    const uint8_t* get_data() const
    {
        return data.data();
    }
    uint8_t* get_data()
    {
        return data.data();
    }

    bool empty() const
    {
        return data.empty();
    }

    template<typename T>
    void set_metadata(T value)
    {
        const size_t metadata_size = sizeof(T);
        const size_t metadata_offset = stride * height;

        if (metadata_size + metadata_offset > data.size())
        {
            throw std::logic_error("Invalid type size");
        }

        auto ptr = reinterpret_cast<T*>(data.data());
        *ptr = value;
    }

    template<typename T>
    const T& get_metadata() const
    {
        const size_t metadata_size = sizeof(T);
        const size_t metadata_offset = stride * height;

        if (metadata_size + metadata_offset > data.size())
        {
            throw std::logic_error("Invalid type size");
        }

        auto ptr = reinterpret_cast<const T*>(data.data());
        return *ptr;
    }

    template<typename T>
    T& get_metadata()
    {
        const size_t metadata_size = sizeof(T);
        const size_t metadata_offset = stride * height;

        if (metadata_size + metadata_offset > data.size())
        {
            throw std::logic_error("Invalid type size");
        }

        auto ptr = reinterpret_cast<T*>(data.data());
        return *ptr;
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(data, width, height, bpp, stride);
    }
private:
    std::vector<uint8_t> data;
    uint32_t width;
    uint32_t height;
    uint32_t bpp;
    uint32_t stride;
};

class image_message: public graph_message
{
    image img;
public:
    image_message()
        : img()
    {}

    void set_image(const image& img)
    {
        this->img = img;
    }
    void set_image(image&& img)
    {
        this->img = std::move(img);
    }
    const image& get_image() const
    {
        return img;
    }
    static std::string get_type()
    {
        return "image";
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(img);
    }
};

CEREAL_REGISTER_TYPE(image_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_message, image_message)

class image_heartbeat_node: public heartbeat_node
{
    image img;
    graph_edge_ptr output;
public:
    image_heartbeat_node()
        : heartbeat_node()
        , img()
        , output(std::make_shared<graph_edge>(this))
    {
        set_output(output);
    }

    void set_image(image img)
    {
        this->img = img;
    }

    const image& get_image() const
    {
        return img;
    }
    
    virtual std::string get_proc_name() const override
    {
        return "image_heartbeat";
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(cereal::base_class<heartbeat_node>(this));
        archive(img);
    }

    virtual void tick() override
    {
        auto msg = std::make_shared<image_message>();
        msg->set_image(img);
        output->send(msg);
    }
};

CEREAL_REGISTER_TYPE(image_heartbeat_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(heartbeat_node, image_heartbeat_node)
