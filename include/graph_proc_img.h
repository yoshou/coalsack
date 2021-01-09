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
        const size_t metadata_offset = 0;

        if (metadata_size + metadata_offset > metadata.size())
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
        const size_t metadata_offset = 0;

        if (metadata_size + metadata_offset > metadata.size())
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
        const size_t metadata_offset = 0;

        if (metadata_size + metadata_offset > metadata.size())
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
    std::vector<uint8_t> metadata;
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

enum class stream_type
{
    ANY,
    DEPTH,
    COLOR,
    INFRERED,
};

enum class stream_format
{
    ANY,
    Z16,
    RGB8,
    BGR8,
    RGBA8,
    BGRA8,
    Y8,
    Y16,
    YUYV,
    UYVY,
};

class stream_profile
{
    int index;
    stream_type type;
    stream_format format;
    int fps;
    int uid;
public:

    stream_profile(stream_type type = stream_type::ANY, int index = -1, stream_format format = stream_format::ANY, int fps = 0, int uid = 0)
        : index(index)
        , type(type)
        , format(format)
        , fps(fps)
        , uid(uid)
    {}

    int get_index() const
    {
        return index;
    }
    void set_index(int index)
    {
        this->index = index;
    }

    stream_type get_type() const
    {
        return type;
    }
    void set_type(stream_type type)
    {
        this->type = type;
    }

    stream_format get_format() const
    {
        return format;
    }
    void set_format(stream_format format)
    {
        this->format = format;
    }

    int get_fps() const
    {
        return fps;
    }
    void set_fps(int fps)
    {
        this->fps = fps;
    }

    int get_unique_id() const
    {
        return uid;
    }
    void set_unique_id(int uid)
    {
        this->uid = uid;
    }

    template<typename Archive>
    void serialize(Archive& archive)
    {
        archive(index, type, format, fps, uid);
    }
};

template<typename T>
class frame_message : public graph_message
{
    using data_type = T;
    using time_type = double;

    data_type data;
    time_type timestamp;
    uint64_t frame_number;
    std::shared_ptr<stream_profile> profile;

public:
    frame_message()
        : data()
    {
    }

    void set_data(const T& data)
    {
        this->data = data;
    }
    void set_data(T&& data)
    {
        this->data = std::move(data);
    }
    T& get_data()
    {
        return data;
    }
    const T& get_data() const
    {
        return data;
    }
    time_type get_timestamp() const
    {
        return timestamp;
    }
    void set_timestamp(time_type value)
    {
        timestamp = value;
    }
    uint64_t get_frame_number() const
    {
        return frame_number;
    }
    void set_frame_number(uint64_t value)
    {
        frame_number = value;
    }
    std::shared_ptr<stream_profile> get_profile() const
    {
        return profile;
    }
    void set_profile(std::shared_ptr<stream_profile> profile)
    {
        this->profile = profile;
    }

    static std::string get_type()
    {
        return std::string(typeid(T).name()) + "_frame";
    }

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(data, timestamp, frame_number, profile);
    }
};

CEREAL_REGISTER_TYPE(frame_message<image>)
CEREAL_REGISTER_POLYMORPHIC_RELATION(graph_message, frame_message<image>)

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
