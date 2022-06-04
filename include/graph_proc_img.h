#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "graph_proc.h"
#include "syncer.h"

namespace coalsack
{
    enum class image_format
    {
        ANY,
        Y8_UINT,
        Y16_UINT,
        R8G8B8_UINT,
        R8G8B8A8_UINT,
        B8G8R8_UINT,
        B8G8R8A8_UINT,
        Z16_UINT,
        YUY2,
        UYVY,
    };

    class image
    {
    public:
        image()
            : data(), metadata(), width(0), height(0), bpp(0), stride(0), format(image_format::ANY)
        {
        }

        image(uint32_t width, uint32_t height, uint32_t bpp, uint32_t stride, uint32_t metadata_capacity = 0)
            : data(stride * height), metadata(metadata_capacity), width(width), height(height), bpp(bpp), stride(stride), format(image_format::ANY)
        {
        }

        image(uint32_t width, uint32_t height, uint32_t bpp, uint32_t stride, const uint8_t *data, uint32_t metadata_capacity = 0)
            : image(width, height, bpp, stride, metadata_capacity)
        {
            std::copy_n(data, stride * height, this->data.begin());
        }

        image(const image &other)
            : data(other.data), metadata(other.metadata), width(other.width), height(other.height), bpp(other.bpp), stride(other.stride), format(other.format)
        {
        }

        image(image &&other)
            : data(std::move(other.data)), metadata(std::move(other.metadata)), width(other.width), height(other.height), bpp(other.bpp), stride(other.stride), format(other.format)
        {
        }

        image &operator=(const image &other)
        {
            data = other.data;
            metadata = other.metadata;
            width = other.width;
            height = other.height;
            bpp = other.bpp;
            stride = other.stride;
            format = other.format;
            return *this;
        }

        image &operator=(image &&other)
        {
            data = std::move(other.data);
            metadata = std::move(other.metadata);
            width = other.width;
            height = other.height;
            bpp = other.bpp;
            stride = other.stride;
            format = other.format;
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

        const uint8_t *get_data() const
        {
            return data.data();
        }
        uint8_t *get_data()
        {
            return data.data();
        }
        image_format get_format() const
        {
            return format;
        }
        void set_format(image_format format)
        {
            this->format = format;
        }

        bool empty() const
        {
            return data.empty();
        }

        template <typename T>
        void set_metadata(T value)
        {
            const size_t metadata_size = sizeof(T);
            const size_t metadata_offset = 0;

            if (metadata_size + metadata_offset > metadata.size())
            {
                throw std::logic_error("Invalid type size");
            }

            auto ptr = reinterpret_cast<T *>(data.data() + metadata_offset);
            *ptr = value;
        }

        template <typename T>
        const T &get_metadata() const
        {
            const size_t metadata_size = sizeof(T);
            const size_t metadata_offset = 0;

            if (metadata_size + metadata_offset > metadata.size())
            {
                throw std::logic_error("Invalid type size");
            }

            auto ptr = reinterpret_cast<const T *>(data.data() + metadata_offset);
            return *ptr;
        }

        template <typename T>
        T &get_metadata()
        {
            const size_t metadata_size = sizeof(T);
            const size_t metadata_offset = 0;

            if (metadata_size + metadata_offset > metadata.size())
            {
                throw std::logic_error("Invalid type size");
            }

            auto ptr = reinterpret_cast<T *>(data.data() + metadata_offset);
            return *ptr;
        }

        const uint8_t *get_metadata() const
        {
            return metadata.data();
        }
        uint8_t *get_metadata()
        {
            return metadata.data();
        }

        uint32_t get_metadata_size() const
        {
            return metadata.size();
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(data, width, height, bpp, stride, format);
        }

    private:
        std::vector<uint8_t> data;
        std::vector<uint8_t> metadata;
        uint32_t width;
        uint32_t height;
        uint32_t bpp;
        uint32_t stride;
        image_format format;
    };

    class image_message : public graph_message
    {
        image img;

    public:
        image_message()
            : img()
        {
        }

        void set_image(const image &img)
        {
            this->img = img;
        }
        void set_image(image &&img)
        {
            this->img = std::move(img);
        }
        const image &get_image() const
        {
            return img;
        }
        static std::string get_type()
        {
            return "image";
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(img);
        }
    };

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
            : index(index), type(type), format(format), fps(fps), uid(uid)
        {
        }

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

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(index, type, format, fps, uid);
        }
    };

    template <typename T>
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
            : data(), timestamp(), frame_number(0), profile()
        {
        }

        void set_data(const T &data)
        {
            this->data = data;
        }
        void set_data(T &&data)
        {
            this->data = std::move(data);
        }
        T &get_data()
        {
            return data;
        }
        const T &get_data() const
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
        void serialize(Archive &archive)
        {
            archive(data, timestamp, frame_number, profile);
        }
    };

    using blob = std::vector<uint8_t>;
    using blob_frame_message = frame_message<blob>;

    using image_frame_message = frame_message<image>;

    class image_heartbeat_node : public heartbeat_node
    {
        image img;
        graph_edge_ptr output;

    public:
        image_heartbeat_node()
            : heartbeat_node(), img(), output(std::make_shared<graph_edge>(this))
        {
            set_output(output);
        }

        void set_image(image img)
        {
            this->img = img;
        }

        const image &get_image() const
        {
            return img;
        }

        virtual std::string get_proc_name() const override
        {
            return "image_heartbeat";
        }

        template <typename Archive>
        void serialize(Archive &archive)
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

    struct approximate_time_sync_config
    {
        using sync_info = approximate_time;
        double interval;

        approximate_time_sync_config()
            : interval(0)
        {
        }

        explicit approximate_time_sync_config(double interval)
            : interval(interval)
        {
        }

        template <typename T>
        approximate_time create_sync_info(std::shared_ptr<frame_message<T>> message)
        {
            return approximate_time(message->get_timestamp(), interval);
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(interval);
        }

        void set_interval(double interval)
        {
            this->interval = interval;
        }
        double get_interval() const
        {
            return interval;
        }
    };

    template <typename T, typename Config = approximate_time_sync_config>
    class sync_node : public graph_node
    {
        using sync_config = Config;
        using sync_info = typename sync_config::sync_info;
        using syncer_type = stream_syncer<graph_message_ptr, std::string, sync_info>;

        syncer_type syncer;
        graph_edge_ptr output;
        sync_config config;

    public:
        sync_node()
            : graph_node(), syncer(), output(std::make_shared<graph_edge>(this)), config()
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "sync_node";
        }

        void set_config(sync_config config)
        {
            this->config = config;
        }
        const sync_config &get_config() const
        {
            return config;
        }
        sync_config &get_config()
        {
            return config;
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(config);
        }

        virtual void run() override
        {
            syncer.start(std::make_shared<typename syncer_type::callback_type>([this](const std::map<std::string, graph_message_ptr> &frames)
                                                                               {
            auto msg = std::make_shared<object_message>();
            for (auto frame : frames)
            {
                msg->add_field(frame.first, frame.second);
            }
            output->send(msg); }));
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            if (auto frame_msg = std::dynamic_pointer_cast<frame_message<T>>(message))
            {
                syncer.sync(input_name, frame_msg, config.create_sync_info(frame_msg));
            }
        }
    };

    using approximate_time_sync_node = sync_node<image, approximate_time_sync_config>;

    class tiling_node : public graph_node
    {
        graph_edge_ptr output;
        std::uint32_t num_cols;
        std::uint32_t num_rows;
        std::set<std::string> names;

    public:
        tiling_node()
            : graph_node(), output(std::make_shared<graph_edge>(this)), num_cols(0), num_rows(0)
        {
            set_output(output);
        }

        virtual std::string get_proc_name() const override
        {
            return "tiling_node";
        }

        void set_num_rows(std::uint32_t value)
        {
            num_rows = value;
        }
        void set_num_cols(std::uint32_t value)
        {
            num_cols = value;
        }
        std::uint32_t get_num_rows() const
        {
            return num_rows;
        }
        std::uint32_t get_num_cols() const
        {
            return num_cols;
        }

        template <typename Archive>
        void serialize(Archive &archive)
        {
            archive(num_rows);
            archive(num_cols);
        }

        void tiling(const std::vector<const image *> &images, image &output)
        {
            if (images.size() == 0)
            {
                return;
            }
            uint32_t tile_width = 0;
            uint32_t tile_height = 0;
            uint32_t tile_bpp = 0;
            uint32_t tile_stride = 0;

            for (const auto image : images)
            {
                if (image)
                {
                    tile_width = image->get_width();
                    tile_height = image->get_height();
                    tile_bpp = image->get_bpp();
                    tile_stride = image->get_stride();
                    break;
                }
            }

            const auto output_stride = tile_width * num_cols * tile_bpp;
            output = image(tile_width * num_cols, tile_height * num_rows, tile_bpp, output_stride);

            for (std::size_t tile_y = 0; tile_y < num_rows; tile_y++)
            {
                for (std::size_t tile_x = 0; tile_x < num_cols; tile_x++)
                {
                    const auto image_idx = tile_y * num_cols + tile_x;
                    if (image_idx >= images.size())
                    {
                        continue;
                    }

                    const auto image = images[image_idx];
                    if (image == nullptr)
                    {
                        continue;
                    }
                    for (std::size_t y = 0; y < tile_height; y++)
                    {
                        std::copy_n(image->get_data() + y * tile_stride, tile_width * tile_bpp,
                                    output.get_data() + output_stride * (tile_y * tile_height + y) + tile_x * tile_width * tile_bpp);
                    }
                }
            }
        }

        virtual void process(std::string input_name, graph_message_ptr message) override
        {
            if (auto obj_msg = std::dynamic_pointer_cast<object_message>(message))
            {
                std::vector<const image *> images;
                std::shared_ptr<stream_profile> profile;
                double timestamp{0.0};
                std::uint64_t frame_number{0};
                for (const auto &[name, field] : obj_msg->get_fields())
                {
                    if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(field))
                    {
                        names.insert(name);
                    }
                }
                for (const auto &name : names)
                {
                    if (const auto iter = obj_msg->get_fields().find(name); iter != obj_msg->get_fields().end())
                    {
                        const auto &field = iter->second;
                        if (auto image_msg = std::dynamic_pointer_cast<frame_message<image>>(field))
                        {
                            const auto &img = image_msg->get_data();
                            images.push_back(&img);
                            profile = image_msg->get_profile();
                            timestamp = image_msg->get_timestamp();
                            frame_number = image_msg->get_frame_number();
                        }
                        else
                        {
                            images.push_back(nullptr);
                        }
                    }
                }

                image img;
                tiling(images, img);

                auto msg = std::make_shared<frame_message<image>>();

                msg->set_data(std::move(img));
                msg->set_profile(profile);
                msg->set_timestamp(timestamp);
                msg->set_frame_number(frame_number);

                output->send(msg);
            }
        }
    };
}

CEREAL_REGISTER_TYPE(coalsack::image_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::image_message)

CEREAL_REGISTER_TYPE(coalsack::blob_frame_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::blob_frame_message)

CEREAL_REGISTER_TYPE(coalsack::image_frame_message)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_message, coalsack::image_frame_message)

CEREAL_REGISTER_TYPE(coalsack::image_heartbeat_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::heartbeat_node, coalsack::image_heartbeat_node)

CEREAL_REGISTER_TYPE(coalsack::approximate_time_sync_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::approximate_time_sync_node)

CEREAL_REGISTER_TYPE(coalsack::tiling_node)
CEREAL_REGISTER_POLYMORPHIC_RELATION(coalsack::graph_node, coalsack::tiling_node)
