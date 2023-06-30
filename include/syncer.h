#pragma once

#include <cstdint>
#include <queue>
#include <mutex>
#include <memory>
#include <map>
#include <vector>
#include <functional>
#include <spdlog/spdlog.h>

namespace coalsack
{
    class approximate_time
    {
        double timestamp;
        double interval;

    public:
        double get_timestamp() const { return timestamp; }
        double get_interval() const { return interval; }

        approximate_time()
            : timestamp(0), interval(0)
        {
        }
        approximate_time(double timestamp, double interval)
            : timestamp(timestamp), interval(interval)
        {
        }
        approximate_time(const approximate_time &src)
            : timestamp(src.timestamp), interval(src.interval)
        {
        }
        approximate_time &operator=(const approximate_time &src)
        {
            timestamp = src.timestamp;
            interval = src.interval;
            return *this;
        }

        bool is_same(const approximate_time &t) const
        {
            auto max_interval = std::max(interval, t.interval);
            return std::abs(timestamp - t.timestamp) <= (max_interval / 2.0);
        }

        bool is_less_than(const approximate_time &t) const
        {
            return timestamp < t.timestamp;
        }

        approximate_time get_next_time() const
        {
            return approximate_time(timestamp + interval, interval);
        }

        bool is_dropped(approximate_time expect_time)
        {
            if (get_timestamp() <= expect_time.get_timestamp())
            {
                return false;
            }
            if (std::abs(expect_time.get_timestamp() - get_timestamp()) < (10 * interval))
            {
                return false;
            }

            return !is_same(expect_time);
        }
    };

    class frame_number
    {
        int64_t number;

    public:
        int64_t get_number() const { return number; }

        frame_number()
            : number(0)
        {
        }
        frame_number(int64_t number)
            : number(number)
        {
        }
        frame_number(const frame_number &src)
            : number(src.number)
        {
        }
        frame_number &operator=(const frame_number &src)
        {
            number = src.number;
            return *this;
        }

        bool is_same(const frame_number &t) const
        {
            return number == t.number;
        }

        bool is_less_than(const frame_number &t) const
        {
            return number < t.number;
        }

        frame_number get_next_time() const
        {
            return frame_number(number + 1);
        }

        bool is_dropped(frame_number expect_time)
        {
            if (get_number() >= expect_time.get_number())
            {
                return false;
            }
            if (std::abs(expect_time.get_number() - get_number()) <= 4)
            {
                return false;
            }

            return true;
        }
    };

    template <typename DataTy, typename IDTy, typename TimeTy = approximate_time>
    struct sync_frame
    {
        using self_type = sync_frame<DataTy, IDTy, TimeTy>;
        using stream_id_type = IDTy;
        using data_type = DataTy;
        using time_type = TimeTy;

        time_type sync_info;
        stream_id_type stream_id;
        data_type data;

        sync_frame()
            : sync_info(), stream_id(), data()
        {
        }

        sync_frame(time_type sync_info, stream_id_type stream_id, const data_type &data)
            : sync_info(sync_info), stream_id(stream_id), data(data)
        {
        }

        sync_frame(const self_type &other)
            : sync_info(other.sync_info), stream_id(other.stream_id), data(other.data)
        {
        }

        self_type &operator=(const self_type &other)
        {
            sync_info = other.sync_info;
            stream_id = other.stream_id;
            data = other.data;
            return *this;
        }
    };

    template <typename DataTy, typename IDTy, typename TimeTy>
    static bool operator<(const sync_frame<DataTy, IDTy, TimeTy> &a, const sync_frame<DataTy, IDTy, TimeTy> &b)
    {
        return a.sync_info.is_less_than(b.sync_info);
    }

    template <typename DataTy, typename IDTy, typename TimeTy>
    static bool operator>(const sync_frame<DataTy, IDTy, TimeTy> &a, const sync_frame<DataTy, IDTy, TimeTy> &b)
    {
        return b.sync_info.is_less_than(a.sync_info);
    }

    template <typename DataTy, typename IDTy>
    struct synced_frame_callback
    {
        using data_type = DataTy;
        using stream_id_type = IDTy;
        using func_type = std::function<void(const std::map<stream_id_type, data_type> &)>;

        func_type func;

        explicit synced_frame_callback(func_type func)
            : func(func)
        {
        }

        void operator()(const std::map<stream_id_type, data_type> &frames)
        {
            if (func)
            {
                func(frames);
            }
        }
    };

    template <typename DataTy, typename IDTy, typename TimeTy = approximate_time>
    class stream_syncer
    {
    public:
        using stream_id_type = IDTy;
        using data_type = DataTy;
        using time_type = TimeTy;

        using frame_type = sync_frame<data_type, stream_id_type, time_type>;
        using frame_queue = std::priority_queue<frame_type, std::vector<frame_type>, std::greater<frame_type>>;
        using callback_type = synced_frame_callback<data_type, stream_id_type>;

        stream_syncer()
        {
        }

        void start(std::shared_ptr<callback_type> callback)
        {
            _callback = callback;
        }

        void sync(stream_id_type id, data_type data, time_type sync_info)
        {
            frame_type frame(sync_info, id, data);
            sync(frame);
        }

    private:
        std::map<stream_id_type, std::shared_ptr<frame_queue>> _frames_queue;
        std::shared_ptr<callback_type> _callback;
        std::map<stream_id_type, time_type> _times;
        std::mutex mtx;
        std::map<stream_id_type, data_type> frames;

        bool skip_missing_stream(time_type base_time, stream_id_type stream_id)
        {
            auto expect_time = _times[stream_id].get_next_time();
            return base_time.is_dropped(expect_time);
        }

        void sync(frame_type frame)
        {
            std::lock_guard<std::mutex> lock(mtx);

            std::shared_ptr<frame_queue> queue;
            auto stream_id = frame.stream_id;

            auto it = _frames_queue.find(stream_id);
            if (it == _frames_queue.end())
            {
                queue.reset(new frame_queue());
                _frames_queue.insert(std::make_pair(stream_id, queue));
                _times.insert(std::make_pair(stream_id, frame.sync_info));
            }
            else
            {
                queue = it->second;
            }

            _times[stream_id] = frame.sync_info;
            queue->push(frame);

            std::vector<frame_type> synced_frames;

            do
            {
                synced_frames.clear();

                std::vector<frame_type> arrived_frames;
                std::vector<stream_id_type> missing_stream_ids;
                for (auto &queue : _frames_queue)
                {
                    if (queue.second->empty())
                    {
                        missing_stream_ids.push_back(queue.first);
                    }
                    else
                    {
                        arrived_frames.push_back(queue.second->top());
                    }
                }

                if (arrived_frames.size() == 0)
                {
                    break;
                }

                frame_type base_frame = arrived_frames[0];
                synced_frames.push_back(arrived_frames[0]);

                for (size_t i = 1; i < arrived_frames.size(); i++)
                {
                    if (arrived_frames[i].sync_info.is_same(base_frame.sync_info))
                    {
                        synced_frames.push_back(arrived_frames[i]);
                    }
                    else if (arrived_frames[i].sync_info.is_less_than(base_frame.sync_info))
                    {
                        synced_frames.clear();
                        synced_frames.push_back(arrived_frames[i]);
                        base_frame = arrived_frames[i];
                    }
                }

                for (auto missing_stream_id : missing_stream_ids)
                {
                    if (!skip_missing_stream(synced_frames[0].sync_info, missing_stream_id))
                    {
                        synced_frames.clear();
                        break;
                    }
                    else
                    {
                        spdlog::warn("Skipped missing stream");
                    }
                }

                for (auto synced_frame : synced_frames)
                {
                    _frames_queue[synced_frame.stream_id]->pop();
                }

                if (synced_frames.size() > 0)
                {
                    frames.clear();
                    for (auto &synced_frame : synced_frames)
                    {
                        frames[synced_frame.stream_id] = synced_frame.data;
                    }

                    if (_callback)
                    {
                        (*_callback)(frames);
                    }
                }
            } while (synced_frames.size() > 0);
        }
    };
}
