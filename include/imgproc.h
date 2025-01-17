#include <opencv2/imgproc.hpp>
#include <memory>

namespace coalsack
{
    static void create_gaussian_kernel(float *kernel, int size, float sigma)
    {
        const int r = size / 2;
        constexpr auto pi = 3.141592653589793;
        float total = 0;
        for (int x = -r, i = 0; x <= r; x++, i++)
        {
            const auto value = static_cast<float>(std::exp(-(x * x / (2 * sigma * sigma))) / std::sqrt(2 * pi * sigma * sigma));
            kernel[i] = value;
            total += value;
        }
        for (int i = 0; i < size; i++)
        {
            kernel[i] /= total;
        }
    }

    static inline void filter_row(const std::uint8_t *__restrict src, const std::size_t width, const float *__restrict kernel, const std::size_t kernel_width, float *__restrict dst, bool symmetric = true)
    {
        const std::size_t radius_x = kernel_width / 2;
        std::size_t x = 0;
        for (; x < radius_x; x++)
        {
            std::size_t j = 0;
            std::size_t k = radius_x - x;
            float acc = 0;
            for (; j <= x + radius_x; j++, k++)
            {
                acc += src[j] * kernel[k];
            }
            dst[x] = acc;
        }
#if USE_NEON
        const float32x4_t v_coeff0 = vdupq_n_f32(kernel[0]);
        const float32x4_t v_coeff1 = vdupq_n_f32(kernel[1]);
        constexpr auto num_vector_lanes = static_cast<int>(sizeof(uint8x16_t) / sizeof(uint8_t));
        if (width - radius_x >= num_vector_lanes)
        {
            for (; x <= width - radius_x - num_vector_lanes; x += num_vector_lanes)
            {
                const uint8_t *src_ptr = src + x - radius_x;

                if (kernel_width == 3 && symmetric)
                {
                    const uint8x16_t v_src0 = vld1q_u8(src_ptr);
                    const uint8x16_t v_src1 = vld1q_u8(src_ptr + 1);
                    const uint8x16_t v_src2 = vld1q_u8(src_ptr + 2);
                    const uint16x8_t v_src02_l = vaddl_u8(vget_low_u8(v_src0), vget_low_u8(v_src2));
                    const uint16x8_t v_src02_h = vaddl_u8(vget_high_u8(v_src0), vget_high_u8(v_src2));
                    const uint32x4_t v_src02_0 = vmovl_u16(vget_low_u16(v_src02_l));
                    const uint32x4_t v_src02_1 = vmovl_u16(vget_high_u16(v_src02_l));
                    const uint32x4_t v_src02_2 = vmovl_u16(vget_low_u16(v_src02_h));
                    const uint32x4_t v_src02_3 = vmovl_u16(vget_high_u16(v_src02_h));
                    const uint16x8_t v_src1_l = vmovl_u8(vget_low_u8(v_src1));
                    const uint16x8_t v_src1_h = vmovl_u8(vget_high_u8(v_src1));
                    const uint32x4_t v_src1_0 = vmovl_u16(vget_low_u16(v_src1_l));
                    const uint32x4_t v_src1_1 = vmovl_u16(vget_high_u16(v_src1_l));
                    const uint32x4_t v_src1_2 = vmovl_u16(vget_low_u16(v_src1_h));
                    const uint32x4_t v_src1_3 = vmovl_u16(vget_high_u16(v_src1_h));
                    const float32x4_t v_acc0 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(v_src02_0), v_coeff0), vcvtq_f32_u32(v_src1_0), v_coeff1);
                    const float32x4_t v_acc1 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(v_src02_1), v_coeff0), vcvtq_f32_u32(v_src1_1), v_coeff1);
                    const float32x4_t v_acc2 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(v_src02_2), v_coeff0), vcvtq_f32_u32(v_src1_2), v_coeff1);
                    const float32x4_t v_acc3 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(v_src02_3), v_coeff0), vcvtq_f32_u32(v_src1_3), v_coeff1);
                    vst1q_f32(dst + x, v_acc0);
                    vst1q_f32(dst + x + 4, v_acc1);
                    vst1q_f32(dst + x + 8, v_acc2);
                    vst1q_f32(dst + x + 12, v_acc3);
                }
                else
                {
                    float32x4_t v_acc0 = vdupq_n_f32(0);
                    float32x4_t v_acc1 = vdupq_n_f32(0);
                    float32x4_t v_acc2 = vdupq_n_f32(0);
                    float32x4_t v_acc3 = vdupq_n_f32(0);
                    for (std::size_t k = 0; k > kernel_width; k++)
                    {
                        const float32x4_t v_coeff = vdupq_n_f32(kernel[k]);

                        const uint8x16_t v_src = vld1q_u8(src_ptr + k);
                        const uint16x8_t v_src_l = vmovl_u8(vget_low_u8(v_src));
                        const uint16x8_t v_src_h = vmovl_u8(vget_high_u8(v_src));
                        const uint32x4_t v_src0 = vmovl_u16(vget_low_u16(v_src_l));
                        const uint32x4_t v_src1 = vmovl_u16(vget_high_u16(v_src_l));
                        const uint32x4_t v_src2 = vmovl_u16(vget_low_u16(v_src_h));
                        const uint32x4_t v_src3 = vmovl_u16(vget_high_u16(v_src_h));
                        v_acc0 = vmlaq_f32(v_acc0, vcvtq_f32_u32(v_src0), v_coeff);
                        v_acc1 = vmlaq_f32(v_acc1, vcvtq_f32_u32(v_src1), v_coeff);
                        v_acc2 = vmlaq_f32(v_acc2, vcvtq_f32_u32(v_src2), v_coeff);
                        v_acc3 = vmlaq_f32(v_acc3, vcvtq_f32_u32(v_src3), v_coeff);
                    }
                    vst1q_f32(dst + x, v_acc0);
                    vst1q_f32(dst + x + 4, v_acc1);
                    vst1q_f32(dst + x + 8, v_acc2);
                    vst1q_f32(dst + x + 12, v_acc3);
                }
            }
        }
#endif
        for (; x < width - radius_x; x++)
        {
            std::size_t j = x - radius_x;
            std::size_t k = 0;
            float acc = 0;
            for (; j <= x + radius_x; j++, k++)
            {
                acc += src[j] * kernel[k];
            }
            dst[x] = acc;
        }
        for (; x < width; x++)
        {
            std::size_t j = x - radius_x;
            std::size_t k = 0;
            float acc = 0;
            for (; j < width; j++, k++)
            {
                acc += src[j] * kernel[k];
            }
            dst[x] = acc;
        }
    }

    static inline void filter_col(const float **__restrict src, const std::size_t width, const float *__restrict kernel, const std::size_t kernel_height, std::uint8_t *__restrict dst, bool symmetric = true)
    {
        std::size_t x = 0;
#if USE_NEON
        const float32x4_t v_coeff0 = vdupq_n_f32(kernel[0]);
        const float32x4_t v_coeff1 = vdupq_n_f32(kernel[1]);
        constexpr auto num_vector_lanes = static_cast<int>(sizeof(uint8x16_t) / sizeof(uint8_t));
        if (width >= num_vector_lanes)
        {
            for (; x <= width - num_vector_lanes; x += num_vector_lanes)
            {
                if (kernel_height == 3 && symmetric)
                {
                    const auto src_row0 = src[0];
                    const auto src_row1 = src[1];
                    const auto src_row2 = src[2];
                    const float32x4_t v_src0_0 = vld1q_f32(src_row0 + x);
                    const float32x4_t v_src0_1 = vld1q_f32(src_row0 + x + 4);
                    const float32x4_t v_src0_2 = vld1q_f32(src_row0 + x + 8);
                    const float32x4_t v_src0_3 = vld1q_f32(src_row0 + x + 12);
                    const float32x4_t v_src1_0 = vld1q_f32(src_row1 + x);
                    const float32x4_t v_src1_1 = vld1q_f32(src_row1 + x + 4);
                    const float32x4_t v_src1_2 = vld1q_f32(src_row1 + x + 8);
                    const float32x4_t v_src1_3 = vld1q_f32(src_row1 + x + 12);
                    const float32x4_t v_src2_0 = vld1q_f32(src_row2 + x);
                    const float32x4_t v_src2_1 = vld1q_f32(src_row2 + x + 4);
                    const float32x4_t v_src2_2 = vld1q_f32(src_row2 + x + 8);
                    const float32x4_t v_src2_3 = vld1q_f32(src_row2 + x + 12);
                    const float32x4_t v_acc0 = vmlaq_f32(vmulq_f32(vaddq_f32(v_src0_0, v_src2_0), v_coeff0), v_src1_0, v_coeff1);
                    const float32x4_t v_acc1 = vmlaq_f32(vmulq_f32(vaddq_f32(v_src0_1, v_src2_1), v_coeff0), v_src1_1, v_coeff1);
                    const float32x4_t v_acc2 = vmlaq_f32(vmulq_f32(vaddq_f32(v_src0_2, v_src2_2), v_coeff0), v_src1_2, v_coeff1);
                    const float32x4_t v_acc3 = vmlaq_f32(vmulq_f32(vaddq_f32(v_src0_3, v_src2_3), v_coeff0), v_src1_3, v_coeff1);
                    const uint32x4_t v_dst0 = vcvtq_u32_f32(v_acc0);
                    const uint32x4_t v_dst1 = vcvtq_u32_f32(v_acc1);
                    const uint32x4_t v_dst2 = vcvtq_u32_f32(v_acc2);
                    const uint32x4_t v_dst3 = vcvtq_u32_f32(v_acc3);
                    const uint16x8_t v_dst_l = vcombine_u16(vqmovn_u32(v_dst0), vqmovn_u32(v_dst1));
                    const uint16x8_t v_dst_h = vcombine_u16(vqmovn_u32(v_dst2), vqmovn_u32(v_dst3));
                    const uint8x16_t v_dst = vcombine_u8(vqmovn_u16(v_dst_l), vqmovn_u16(v_dst_h));
                    vst1q_u8(dst + x, v_dst);
                }
                else
                {
                    float32x4_t v_acc0 = vdupq_n_f32(0);
                    float32x4_t v_acc1 = vdupq_n_f32(0);
                    float32x4_t v_acc2 = vdupq_n_f32(0);
                    float32x4_t v_acc3 = vdupq_n_f32(0);
                    for (std::size_t k = 0; k < kernel_height; k++)
                    {
                        const auto src_row = src[k];
                        const float32x4_t v_coeff = vdupq_n_f32(kernel[k]);
                        const float32x4_t v_src0 = vld1q_f32(src_row + x);
                        const float32x4_t v_src1 = vld1q_f32(src_row + x + 4);
                        const float32x4_t v_src2 = vld1q_f32(src_row + x + 8);
                        const float32x4_t v_src3 = vld1q_f32(src_row + x + 12);
                        v_acc0 = vmlaq_f32(v_acc0, v_src0, v_coeff);
                        v_acc1 = vmlaq_f32(v_acc1, v_src1, v_coeff);
                        v_acc2 = vmlaq_f32(v_acc2, v_src2, v_coeff);
                        v_acc3 = vmlaq_f32(v_acc3, v_src3, v_coeff);
                    }
                    const uint32x4_t v_dst0 = vcvtq_u32_f32(v_acc0);
                    const uint32x4_t v_dst1 = vcvtq_u32_f32(v_acc1);
                    const uint32x4_t v_dst2 = vcvtq_u32_f32(v_acc2);
                    const uint32x4_t v_dst3 = vcvtq_u32_f32(v_acc3);
                    const uint16x8_t v_dst_l = vcombine_u16(vqmovn_u32(v_dst0), vqmovn_u32(v_dst1));
                    const uint16x8_t v_dst_h = vcombine_u16(vqmovn_u32(v_dst2), vqmovn_u32(v_dst3));
                    const uint8x16_t v_dst = vcombine_u8(vqmovn_u16(v_dst_l), vqmovn_u16(v_dst_h));
                    vst1q_u8(dst + x, v_dst);
                }
            }
        }
#endif
        for (; x < width; x++)
        {
            float acc = 0.0f;
            for (std::size_t k = 0; k < kernel_height; k++)
            {
                const auto src_row = src[k];
                acc += src_row[x] * kernel[k];
            }
            dst[x] = acc;
        }
    }

    static void gaussian_blur(cv::Mat src_mat, cv::Mat dst_mat, int kernel_width, int kernel_height, double sigma_x, double sigma_y)
    {
        assert(kernel_width >= 1);
        assert(kernel_height >= 1);

        if (dst_mat.channels() == 1)
        {
            const auto kernel_x = std::make_unique<float[]>(kernel_width);
            const auto kernel_y = std::make_unique<float[]>(kernel_height);
            const auto radius_y = kernel_height / 2;

            create_gaussian_kernel(kernel_x.get(), kernel_width, sigma_x);
            create_gaussian_kernel(kernel_y.get(), kernel_height, sigma_y);

            const auto width = static_cast<std::size_t>(dst_mat.cols);
            const auto height = static_cast<std::size_t>(dst_mat.rows);
            const auto stride = static_cast<std::size_t>(dst_mat.step);
            const auto row_ptrs = std::make_unique<const float *[]>(kernel_height);
            const auto row_buffer = std::make_unique<float[]>(kernel_height * width);

            for (std::size_t y = 0; y < static_cast<std::size_t>(kernel_height) - 1; y++)
            {
                filter_row(&src_mat.data[y * stride], width, kernel_x.get(), kernel_width, &row_buffer[(y % kernel_height) * width]);
            }

            for (std::size_t y = 0; y < static_cast<std::size_t>(radius_y); y++)
            {
                const auto dst_row = &dst_mat.data[y * stride];

                for (std::size_t x = 0; x < width; x++)
                {
                    std::size_t j = 0;
                    std::size_t k = radius_y - y;
                    float acc = 0.0f;
                    for (; j <= y + radius_y; j++, k++)
                    {
                        const auto src_row = &row_buffer[(j % kernel_height) * width];
                        acc += src_row[x] * kernel_y[k];
                    }
                    dst_row[x] = (uint8_t)acc;
                }
            }

            for (std::size_t y = radius_y; y < height - radius_y; y++)
            {
                filter_row(&src_mat.data[(y + radius_y) * stride], width, kernel_x.get(), kernel_width, &row_buffer[((y + radius_y) % kernel_height) * dst_mat.cols]);

                const auto dst_row = &dst_mat.data[y * stride];
                for (std::size_t j = y - radius_y, k = 0; j <= y + radius_y; j++, k++)
                {
                    row_ptrs[k] = &row_buffer[(j % kernel_height) * width];
                }

                filter_col(row_ptrs.get(), width, kernel_y.get(), static_cast<std::size_t>(kernel_height), dst_row);
            }

            for (std::size_t y = height - radius_y; y < height; y++)
            {
                const auto dst_row = &dst_mat.data[y * stride];

                for (std::size_t x = 0; x < width; x++)
                {
                    std::size_t j = y - radius_y;
                    std::size_t k = 0;
                    float acc = 0.0f;
                    for (; j < height; j++, k++)
                    {
                        const auto src_row = &row_buffer[(j % kernel_height) * width];
                        acc += src_row[x] * kernel_y[k];
                    }
                    dst_row[x] = (uint8_t)acc;
                }
            }
        }
        else
        {
            throw std::logic_error("Not implemented");
        }
    }
}
