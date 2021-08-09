#include <gtest/gtest.h>
#include <numeric>
#include "syncer.h"

TEST(SyncerTest, Test1)
{
    using syncer_type = stream_syncer<int, int>;
    syncer_type syncer;

    int fps = 90;
    double interval = 1000.0 / fps;

    int num_stream = 10;

    // First data outputs immediately because there are no other streams you need to wait. 
    syncer.sync(0, 0, approximate_time(0, interval));

    const auto frame1_time = 5;
    const auto frame2_time = 6;

    // The other stream are blocked to wait stream 0.
    for (int i = 1; i < num_stream; i++)
    {
        syncer.sync(i, 0, approximate_time(interval * frame1_time, interval));
    }

    // Old data is output
    syncer.sync(0, 0, approximate_time(interval * frame2_time, interval));

    syncer.start(std::make_shared<typename syncer_type::callback_type>([&](const std::map<int, int>& frames) {
        std::vector<int> expected(num_stream);
        std::iota(expected.begin(), expected.end(), 0);

        std::vector<int> result;
        for (auto [id, data] : frames)
        {
            result.push_back(id);
        }
        std::sort(result.begin(), result.end());

        ASSERT_EQ(result.size(), expected.size());
        for (std::size_t i = 0; i < result.size(); i++)
        {
            ASSERT_EQ(result[i], expected[i]);
        }
    }));

    // All stream data ready
    for (int i = 1; i < num_stream; i++)
    {
        syncer.sync(i, i, approximate_time(interval * frame2_time, interval));
    }
}