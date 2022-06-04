#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

struct u32vec2
{
    std::uint32_t x;
    std::uint32_t y;
};

struct f32vec2
{
    float x;
    float y;

    f32vec2 &operator+=(const f32vec2 &v)
    {
        x += v.x;
        y += v.y;
        return *this;
    }
    f32vec2 &operator-=(const f32vec2 &v)
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }
    f32vec2 &operator*=(const f32vec2 &v)
    {
        x *= v.x;
        y *= v.y;
        return *this;
    }
    f32vec2 &operator*=(float s)
    {
        x *= s;
        y *= s;
        return *this;
    }
};

static f32vec2 operator+(const f32vec2 &left, const f32vec2 &right)
{
    return f32vec2{left.x + right.x, left.y + right.y};
}
static f32vec2 operator-(const f32vec2 &left, const f32vec2 &right)
{
    return f32vec2{left.x - right.x, left.y - right.y};
}
static f32vec2 operator*(const f32vec2 &left, const f32vec2 &right)
{
    return f32vec2{left.x * right.x, left.y * right.y};
}
static f32vec2 operator*(const f32vec2 &v, float s)
{
    return f32vec2{v.x * s, v.y * s};
}

static float norm(const f32vec2 &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

class contours_detector
{
    const std::uint8_t *image;
    std::size_t width;
    std::size_t height;

public:
    contours_detector(const std::uint8_t *image, std::size_t width, std::size_t height)
        : image(image), width(width), height(height)
    {
    }

private:
    static std::size_t find_nonzero(const std::uint8_t *__restrict image, std::size_t length)
    {
        std::size_t x = 0;
#if USE_NEON
        constexpr auto num_vector_lanes = sizeof(uint8x16_t) / sizeof(uint8_t);
        for (; x <= length - num_vector_lanes; x += num_vector_lanes)
        {
            uint8x16_t v_src = vld1q_u8(image + x);
            if (vmaxvq_u8(v_src) != 0)
            {
                break;
            }
        }
#endif
        for (; x < length; x++)
        {
            if (image[x] != 0)
            {
                return x;
            }
        }

        return length;
    }

    static std::size_t find_zero(const std::uint8_t *image, std::size_t length)
    {
        std::size_t x = 0;
#if USE_NEON
        constexpr auto num_vector_lanes = sizeof(uint8x16_t) / sizeof(uint8_t);
        for (; x <= length - num_vector_lanes; x += num_vector_lanes)
        {
            uint8x16_t v_src = vld1q_u8(image + x);
            if (vminvq_u8(v_src) == 0)
            {
                break;
            }
        }
#endif
        for (; x < length; x++)
        {
            if (image[x] == 0)
            {
                return x;
            }
        }
        return length;
    }

    struct linked_point
    {
        std::size_t x;
        std::size_t y;
        std::size_t link;
    };

    enum class CONNECTING_LINE
    {
        NONE,
        UPPER,
        LOWER,
    };

public:
    void detect(std::vector<std::vector<u32vec2>> &contours)
    {
        std::vector<linked_point> points;

        std::vector<std::size_t> internal_contour_heads;
        std::vector<std::size_t> external_contour_heads;

        for (std::size_t x = 0; x < width;)
        {
            x += find_nonzero(image + x, width - x);

            if (x >= width)
            {
                break;
            }

            linked_point start_point;
            start_point.x = x;
            start_point.y = 0;
            start_point.link = points.size() + 1;
            points.push_back(start_point);

            x += 1;
            x += find_zero(image + x, width - x);

            linked_point end_point;
            end_point.x = x - 1;
            end_point.y = 0;
            end_point.link = points.size() - 1;
            points.push_back(end_point);

            external_contour_heads.push_back(points.size() - 1);
        }

        std::size_t upper_row_points_start = 0;
        std::size_t upper_row_points_end = points.size();
        for (std::size_t y = 1; y < height; y++)
        {
            for (std::size_t x = 0; x < width;)
            {
                x += find_nonzero(image + y * width + x, width - x);

                if (x >= width)
                {
                    break;
                }

                linked_point start_point;
                start_point.x = x;
                start_point.y = y;
                start_point.link = points.size() + 1;
                points.push_back(start_point);

                x += find_zero(image + y * width + x, width - x);

                linked_point end_point;
                end_point.x = x - 1;
                end_point.y = y;
                end_point.link = points.size() - 1;
                points.push_back(end_point);
            }

            std::size_t lower_row_points_start = upper_row_points_end;
            std::size_t lower_row_points_end = points.size();

            std::size_t i = upper_row_points_start;
            std::size_t j = lower_row_points_start;
            CONNECTING_LINE connecting_line = CONNECTING_LINE::NONE;
            std::size_t prev_point = -1;
            while (i < upper_row_points_end && j < lower_row_points_end)
            {
                auto &upper_segment_start = points[i];
                auto &upper_segment_end = points[i + 1];
                auto &lower_segment_start = points[j];
                auto &lower_segment_end = points[j + 1];

                if (connecting_line == CONNECTING_LINE::NONE)
                {
                    if (upper_segment_end.x < lower_segment_end.x)
                    {
                        if (upper_segment_end.x >= lower_segment_start.x - 1)
                        {
                            // Connecting a -> b
                            //   b---------
                            //       a-----------
                            lower_segment_start.link = i;

                            // Check upper line to terminate end
                            connecting_line = CONNECTING_LINE::UPPER;
                            prev_point = i + 1;
                        }
                        else
                        {
                            // Not overlapping
                        }
                        i += 2;
                    }
                    else
                    {
                        // Connecting a -> b
                        //           b-----------
                        //      a-------------
                        if (upper_segment_start.x <= lower_segment_end.x + 1)
                        {
                            lower_segment_start.link = i;

                            // Check lower line to terminate end
                            connecting_line = CONNECTING_LINE::LOWER;
                            prev_point = j + 1;
                        }
                        else
                        {
                            // Not overlapping
                            external_contour_heads.push_back(j);
                        }
                        j += 2;
                    }
                }
                else if (connecting_line == CONNECTING_LINE::UPPER)
                {
                    if (upper_segment_start.x > lower_segment_end.x + 1)
                    {
                        // Not overlapping
                        // Terminate link
                        // Connecting c -> d
                        //   b---------c
                        //       a-----------d
                        points[prev_point].link = j + 1;
                        connecting_line = CONNECTING_LINE::NONE;
                        j += 2;
                    }
                    else
                    {
                        // Connecting c -> d
                        //   b-----c   d---
                        //       a-----------
                        points[prev_point].link = i;

                        if (upper_segment_end.x < lower_segment_end.x)
                        {
                            // Check upper line to terminate end
                            prev_point = i + 1;
                            i += 2;
                        }
                        else
                        {
                            // Check lower line to terminate end
                            connecting_line = CONNECTING_LINE::LOWER;
                            prev_point = j + 1;
                            j += 2;
                        }
                    }
                }
                else if (connecting_line == CONNECTING_LINE::LOWER)
                {
                    if (lower_segment_start.x > upper_segment_end.x + 1)
                    {
                        // Not overlapping
                        // Terminate link
                        // Connecting c -> d
                        //        b---------c
                        //   a-----------d
                        upper_segment_end.link = prev_point;
                        connecting_line = CONNECTING_LINE::NONE;
                        i += 2;
                    }
                    else
                    {
                        // Connecting c -> d
                        //        b-----------
                        //   a--------d  c----
                        lower_segment_start.link = prev_point;

                        internal_contour_heads.push_back(j);

                        if (lower_segment_end.x < upper_segment_end.x)
                        {
                            // Check lower line to terminate end
                            prev_point = j + 1;
                            j += 2;
                        }
                        else
                        {
                            // Check upper line to terminate end
                            connecting_line = CONNECTING_LINE::UPPER;
                            prev_point = i + 1;
                            i += 2;
                        }
                    }
                }
            }

            for (; j < lower_row_points_end; j += 2)
            {
                auto &lower_segment_start = points[j];
                auto &lower_segment_end = points[j + 1];

                if (connecting_line != CONNECTING_LINE::NONE)
                {
                    // Terminate link
                    points[prev_point].link = j + 1;
                    connecting_line = CONNECTING_LINE::NONE;
                }
                else
                {
                    // Not overlapping
                    external_contour_heads.push_back(j);
                }
            }

            for (; i < upper_row_points_end; i += 2)
            {
                auto &upper_segment_start = points[i];
                auto &upper_segment_end = points[i + 1];

                if (connecting_line != CONNECTING_LINE::NONE)
                {
                    // Terminate link
                    upper_segment_end.link = prev_point;
                    connecting_line = CONNECTING_LINE::NONE;
                }
            }

            upper_row_points_start = lower_row_points_start;
            upper_row_points_end = lower_row_points_end;
        }

        const auto visited = std::make_unique<bool[]>(points.size());
        for (std::size_t i = 0; i < external_contour_heads.size(); i++)
        {
            const auto start = external_contour_heads[i];
            if (visited[start])
            {
                continue;
            }

            auto idx = start;
            std::vector<u32vec2> contour;
            do
            {
                contour.push_back(u32vec2{
                    static_cast<std::uint32_t>(points[idx].x), static_cast<std::uint32_t>(points[idx].y)});
                visited[idx] = true;
                idx = points[idx].link;
            } while (idx != start);

            contours.push_back(contour);
        }

        for (std::size_t i = 0; i < internal_contour_heads.size(); i++)
        {
            const auto start = internal_contour_heads[i];
            if (visited[start])
            {
                continue;
            }

            auto idx = start;
            std::vector<u32vec2> contour;
            do
            {
                contour.push_back(u32vec2{
                    static_cast<std::uint32_t>(points[idx].x), static_cast<std::uint32_t>(points[idx].y)});
                visited[idx] = true;
                idx = points[idx].link;
            } while (idx != start);

            contours.push_back(contour);
        }
    }
};

static void threshold(const std::uint8_t *__restrict src, std::size_t length,
                      std::uint8_t threshold, std::uint8_t max_value, std::uint8_t *__restrict dst)
{
    std::size_t x = 0;
#if USE_NEON
    constexpr auto num_vector_lanes = sizeof(uint8x16_t) / sizeof(uint8_t);

    uint8x16_t v_threshold = vdupq_n_u8(threshold);
    uint8x16_t v_max_value = vdupq_n_u8(max_value);
    for (; x <= length - num_vector_lanes; x += num_vector_lanes)
    {
        uint8x16_t v_src = vld1q_u8(src + x);
        uint8x16_t v_mask = vcgtq_u8(v_src, v_threshold);
        uint8x16_t v_dst = vandq_u8(v_mask, v_max_value);
        vst1q_u8(dst + x, v_dst);
    }
#endif
    for (; x < length; x++)
    {
        dst[x] = (src[x] > threshold) ? max_value : 0;
    }
}

struct moments_t
{
    double m00;
    double m10;
    double m01;
    double m20;
    double m11;
    double m02;
    double m30;
    double m21;
    double m12;
    double m03;

    double mu20;
    double mu11;
    double mu02;
    double mu30;
    double mu21;
    double mu12;
    double mu03;

    double nu20;
    double nu11;
    double nu02;
    double nu30;
    double nu21;
    double nu12;
    double nu03;
};

static void update_central_moments(moments_t &moments)
{
    double cx = 0;
    double cy = 0;
    double mu20;
    double mu11;
    double mu02;
    double inv_m00 = 0.0;
    if (std::abs(moments.m00) > std::numeric_limits<decltype(moments.m00)>::epsilon())
    {
        inv_m00 = 1. / moments.m00;
        cx = moments.m10 * inv_m00;
        cy = moments.m01 * inv_m00;
    }

    // mu20 = m20 - m10*cx
    mu20 = moments.m20 - moments.m10 * cx;
    // mu11 = m11 - m10*cy
    mu11 = moments.m11 - moments.m10 * cy;
    // mu02 = m02 - m01*cy
    mu02 = moments.m02 - moments.m01 * cy;

    moments.mu20 = mu20;
    moments.mu11 = mu11;
    moments.mu02 = mu02;

    // mu30 = m30 - cx*(3*mu20 + cx*m10)
    moments.mu30 = moments.m30 - cx * (3 * mu20 + cx * moments.m10);
    mu11 += mu11;
    // mu21 = m21 - cx*(2*mu11 + cx*m01) - cy*mu20
    moments.mu21 = moments.m21 - cx * (mu11 + cx * moments.m01) - cy * mu20;
    // mu12 = m12 - cy*(2*mu11 + cy*m10) - cx*mu02
    moments.mu12 = moments.m12 - cy * (mu11 + cy * moments.m10) - cx * mu02;
    // mu03 = m03 - cy*(3*mu02 + cy*m01)
    moments.mu03 = moments.m03 - cy * (3 * mu02 + cy * moments.m01);

    double inv_sqrt_m00 = std::sqrt(std::abs(inv_m00));
    double s2 = inv_m00 * inv_m00, s3 = s2 * inv_sqrt_m00;

    moments.nu20 = moments.mu20 * s2;
    moments.nu11 = moments.mu11 * s2;
    moments.nu02 = moments.mu02 * s2;
    moments.nu30 = moments.mu30 * s3;
    moments.nu21 = moments.mu21 * s3;
    moments.nu12 = moments.mu12 * s3;
    moments.nu03 = moments.mu03 * s3;
}

static inline moments_t compute_contour_moments(const std::vector<u32vec2> &contour)
{
    moments_t m = {};

    if (contour.size() == 0)
    {
        return m;
    }

    double a00 = 0;
    double a10 = 0;
    double a01 = 0;
    double a20 = 0;
    double a11 = 0;
    double a02 = 0;
    double a30 = 0;
    double a21 = 0;
    double a12 = 0;
    double a03 = 0;

    auto xi_1 = static_cast<double>(contour.back().x);
    auto yi_1 = static_cast<double>(contour.back().y);

    auto xi_12 = xi_1 * xi_1;
    auto yi_12 = yi_1 * yi_1;

    for (const auto &pt : contour)
    {
        const auto xi = static_cast<double>(pt.x);
        const auto yi = static_cast<double>(pt.y);

        const auto xi2 = xi * xi;
        const auto yi2 = yi * yi;
        const auto dxy = xi_1 * yi - xi * yi_1;
        const auto xii_1 = xi_1 + xi;
        const auto yii_1 = yi_1 + yi;

        a00 += dxy;
        a10 += dxy * xii_1;
        a01 += dxy * yii_1;
        a20 += dxy * (xi_1 * xii_1 + xi2);
        a11 += dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi));
        a02 += dxy * (yi_1 * yii_1 + yi2);
        a30 += dxy * xii_1 * (xi_12 + xi2);
        a03 += dxy * yii_1 * (yi_12 + yi2);
        a21 += dxy * (xi_12 * (3 * yi_1 + yi) + 2 * xi * xi_1 * yii_1 +
                      xi2 * (yi_1 + 3 * yi));
        a12 += dxy * (yi_12 * (3 * xi_1 + xi) + 2 * yi * yi_1 * xii_1 +
                      yi2 * (xi_1 + 3 * xi));

        xi_1 = xi;
        yi_1 = yi;
        xi_12 = xi2;
        yi_12 = yi2;
    }

    if (std::abs(a00) > std::numeric_limits<decltype(a00)>::epsilon())
    {
        double db1_2, db1_6, db1_12, db1_24, db1_20, db1_60;

        if (a00 > 0)
        {
            db1_2 = 0.5;
            db1_6 = 0.16666666666666666666666666666667;
            db1_12 = 0.083333333333333333333333333333333;
            db1_24 = 0.041666666666666666666666666666667;
            db1_20 = 0.05;
            db1_60 = 0.016666666666666666666666666666667;
        }
        else
        {
            db1_2 = -0.5;
            db1_6 = -0.16666666666666666666666666666667;
            db1_12 = -0.083333333333333333333333333333333;
            db1_24 = -0.041666666666666666666666666666667;
            db1_20 = -0.05;
            db1_60 = -0.016666666666666666666666666666667;
        }

        m.m00 = a00 * db1_2;
        m.m10 = a10 * db1_6;
        m.m01 = a01 * db1_6;
        m.m20 = a20 * db1_12;
        m.m11 = a11 * db1_24;
        m.m02 = a02 * db1_12;
        m.m30 = a30 * db1_20;
        m.m21 = a21 * db1_60;
        m.m12 = a12 * db1_60;
        m.m03 = a03 * db1_20;
    }
    return m;
}

float compute_arc_length(const std::vector<u32vec2> &contour)
{
    float perimeter = 0;

    int i;

    if (contour.size() < 2)
    {
        return 0.;
    }

    auto prev = f32vec2{
        static_cast<float>(contour.back().x),
        static_cast<float>(contour.back().y),
    };

    for (const auto &p : contour)
    {
        const auto pt = f32vec2{
            static_cast<float>(p.x),
            static_cast<float>(p.y),
        };
        const auto dx = pt.x - prev.x;
        const auto dy = pt.y - prev.y;
        perimeter += std::sqrt(dx * dx + dy * dy);

        prev = pt;
    }

    return perimeter;
}

struct circle_t
{
    f32vec2 p;
    float r;
};

class blob_detector
{
    const std::uint8_t *image;
    std::size_t width;
    std::size_t height;

public:
    blob_detector(const std::uint8_t *image, std::size_t width, std::size_t height)
        : image(image), width(width), height(height), min_threshold(50.0), max_threshold(220.0), step_threshold(10.0), min_area(25.0), max_area(5000.0), min_circularity(0.8f), max_circularity(std::numeric_limits<float>::max()), min_dist_between_blobs(10), min_repeatability(2)
    {
    }

    double min_threshold;
    double max_threshold;
    double step_threshold;
    double min_area;
    double max_area;
    double min_circularity;
    double max_circularity;
    float min_dist_between_blobs;
    std::size_t min_repeatability;

    void detect(std::vector<circle_t> &keypoints)
    {
        std::vector<std::vector<circle_t>> centers;
        const auto binarized_image = std::make_unique<std::uint8_t[]>(width * height);
        for (double thresh = min_threshold; thresh < max_threshold; thresh += step_threshold)
        {
            threshold(image, width * height, static_cast<std::uint8_t>(thresh), 255, binarized_image.get());

            std::vector<std::vector<u32vec2>> contours;
            contours_detector detector(binarized_image.get(), width, height);
            detector.detect(contours);

            std::vector<circle_t> circles;
            for (std::size_t i = 0; i < contours.size(); i++)
            {
                const auto moments = compute_contour_moments(contours[i]);

                if (moments.m00 == 0.0)
                {
                    continue;
                }

                const auto area = moments.m00;
                if (area < min_area || area >= max_area)
                {
                    continue;
                }

                const auto perimeter = compute_arc_length(contours[i]);
                double ratio = 4.0 * M_PI * area / (perimeter * perimeter);
                if (ratio < min_circularity || ratio >= max_circularity)
                {
                    continue;
                }

                const auto center = f32vec2{
                    static_cast<float>(moments.m10 / moments.m00),
                    static_cast<float>(moments.m01 / moments.m00)};

                std::vector<float> dists;
                for (const auto &pt : contours[i])
                {
                    const auto x = pt.x - center.x;
                    const auto y = pt.y - center.y;
                    dists.push_back(std::sqrt(x * x + y * y));
                }
                std::sort(dists.begin(), dists.end());
                const auto radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.f;

                circles.push_back(circle_t{
                    center, radius});
            }

            std::vector<std::vector<circle_t>> new_centers;
            for (size_t i = 0; i < circles.size(); i++)
            {
                bool is_new = true;
                for (size_t j = 0; j < centers.size(); j++)
                {
                    double dist = norm(centers[j][centers[j].size() / 2].p - circles[i].p);
                    is_new = dist >= min_dist_between_blobs && dist >= centers[j][centers[j].size() / 2].r && dist >= circles[i].r;
                    if (!is_new)
                    {
                        centers[j].push_back(circles[i]);

                        size_t k = centers[j].size() - 1;
                        while (k > 0 && circles[i].r < centers[j][k - 1].r)
                        {
                            centers[j][k] = centers[j][k - 1];
                            k--;
                        }
                        centers[j][k] = circles[i];

                        break;
                    }
                }
                if (is_new)
                {
                    new_centers.push_back(std::vector<circle_t>(1, circles[i]));
                }
            }
            std::copy(new_centers.begin(), new_centers.end(), std::back_inserter(centers));
        }

        for (std::size_t i = 0; i < centers.size(); i++)
        {
            if (centers[i].size() < min_repeatability)
            {
                continue;
            }

            auto sum_point = f32vec2{0.f, 0.f};
            float normalizer = 0;
            for (size_t j = 0; j < centers[i].size(); j++)
            {
                sum_point += centers[i][j].p;
                normalizer += 1.f;
            }
            sum_point *= (1.f / normalizer);
            keypoints.push_back(circle_t{
                sum_point, centers[i][centers[i].size() / 2].r});
        }
    }
};
