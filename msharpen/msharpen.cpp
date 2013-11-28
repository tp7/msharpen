#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include "avisynth.h"
#include <math.h>
#include <malloc.h>
#include <emmintrin.h>
#include <stdint.h>
#include <algorithm>


inline bool is_ptr_aligned(const void *ptr, size_t align) {
    return (((uintptr_t)ptr & ((uintptr_t)(align-1))) == 0);
}

static void planar_blur_c(uint8_t *dstp, const uint8_t *srcp, int dst_pitch, int src_pitch, int height, int width) {
    memcpy(dstp, srcp, width);

    dstp += dst_pitch;
    srcp += src_pitch;

    for (int y = 1; y < height-1; ++y) {
        dstp[0] = srcp[0];
        for (int x = 1; x < width-1; ++x) {
            int c1 = srcp[x-src_pitch-1] + srcp[x-1] + srcp[x+src_pitch-1];
            int c2 = srcp[x-src_pitch] + srcp[x] + srcp[x+src_pitch];
            int c3 = srcp[x-src_pitch+1] + srcp[x+1] + srcp[x+src_pitch+1];
            dstp[x] = (c1+c2+c3+4) / 9;
        }
        dstp[width-1] = srcp[width-1];
        srcp += src_pitch;
        dstp += dst_pitch;
    }

    memcpy(dstp, srcp, width);
}

__forceinline __m128i removegrain_mode20_sse2(const uint8_t* pSrc, int srcPitch) {
    __m128i a1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrc-srcPitch-1));
    __m128i a2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrc-srcPitch));
    __m128i a3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrc-srcPitch+1));
    __m128i a4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrc-1));
    __m128i c = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrc));
    __m128i a5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrc+1));
    __m128i a6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrc+srcPitch-1));
    __m128i a7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrc+srcPitch));
    __m128i a8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pSrc+srcPitch+1));

    auto zero = _mm_setzero_si128();
    auto onenineth = _mm_set1_epi16((uint16_t)(((1u << 16) + 4) / 9));
    auto bias = _mm_set1_epi16(4);

    auto a1unpck_lo = _mm_unpacklo_epi8(a1, zero);
    auto a2unpck_lo = _mm_unpacklo_epi8(a2, zero);
    auto a3unpck_lo = _mm_unpacklo_epi8(a3, zero);
    auto a4unpck_lo = _mm_unpacklo_epi8(a4, zero);
    auto a5unpck_lo = _mm_unpacklo_epi8(a5, zero);
    auto a6unpck_lo = _mm_unpacklo_epi8(a6, zero);
    auto a7unpck_lo = _mm_unpacklo_epi8(a7, zero);
    auto a8unpck_lo = _mm_unpacklo_epi8(a8, zero);
    auto cunpck_lo = _mm_unpacklo_epi8(c, zero);

    auto sum_t1 = _mm_adds_epu16(a1unpck_lo, a2unpck_lo);
    sum_t1 = _mm_adds_epu16(sum_t1, a3unpck_lo);
    sum_t1 = _mm_adds_epu16(sum_t1, a4unpck_lo);

    auto sum_t2 = _mm_adds_epu16(a5unpck_lo, a6unpck_lo);
    sum_t2 = _mm_adds_epu16(sum_t2, a7unpck_lo);
    sum_t2 = _mm_adds_epu16(sum_t2, a8unpck_lo);

    auto sum = _mm_adds_epu16(sum_t1, sum_t2);
    sum = _mm_adds_epu16(sum, cunpck_lo);
    sum = _mm_adds_epu16(sum, bias);

    auto result_lo = _mm_mulhi_epu16(sum, onenineth);

    return _mm_packus_epi16(result_lo, zero);
}

static void planar_blur_sse2(uint8_t *dstp, const uint8_t *srcp, int dst_pitch, int src_pitch, int height, int width) {
    memcpy(dstp, srcp, width);

    srcp += src_pitch;
    dstp += dst_pitch;
    int mod8_width = width / 8 * 8;

    for (int y = 1; y < height-1; ++y) {
        dstp[0] = srcp[0];

        for (int x = 1; x < mod8_width-1; x += 8) {
            __m128i result = removegrain_mode20_sse2(srcp+x, src_pitch);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp+x), result);
        }

        if (mod8_width != width) {
            __m128i result = removegrain_mode20_sse2(srcp+width-9, src_pitch);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp+width-9), result);
        }

        dstp[width-1] = srcp[width-1];

        srcp += src_pitch;
        dstp += dst_pitch;
    }

    memcpy(dstp, srcp, width);
}

static __forceinline __m128i abs_diff(__m128i a, __m128i b) {
    auto positive = _mm_subs_epu8(a, b);
    auto negative = _mm_subs_epu8(b, a);
    return _mm_or_si128(positive, negative);
}

static __forceinline __m128i planar_detect_edges_sse2_core(const __m128i &src, const __m128i &srcn, const __m128i &srcnp, const __m128i &threshold, const __m128i &ff) {
    auto adiff1 = abs_diff(src, srcn);
    auto mask1 = _mm_subs_epu8(adiff1, threshold);
    auto adiff2 = abs_diff(src, srcnp);
    auto mask2 = _mm_subs_epu8(adiff2, threshold);
    auto mask = _mm_or_si128(mask1, mask2);
    
    auto leq_thr = _mm_cmpeq_epi8(mask, _mm_setzero_si128());
    return _mm_xor_si128(leq_thr, ff);
}

static void planar_detect_edges_sse2(uint8_t *dstp, const uint8_t *srcp, int dst_pitch, int src_pitch, int height, int width, int threshold, bool mask) {
    threshold = std::max(1, threshold);
    int loop_limit = std::min(width+15, dst_pitch);

    __m128i thresh_vector = _mm_set1_epi8(threshold);
#pragma warning(disable: 4309)
    __m128i ff = _mm_set1_epi8(0xFF);
    __m128i left_mask  = _mm_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0);
#pragma warning(default: 4309)

    const uint8_t *srcpn = srcp+src_pitch;
    memset(dstp, 0, height * dst_pitch);

    for (int y = 0; y < height-1; y++)
    {
        auto src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp));
        auto srcn = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcpn));
        auto srcnp = _mm_slli_si128(srcn, 2);
        auto dst = planar_detect_edges_sse2_core(src, srcn, srcnp, thresh_vector, ff);
        dst = _mm_and_si128(dst, left_mask);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp), dst);

        for (int x = 16; x < loop_limit; x+=16) {
            auto src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x));
            auto srcn = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcpn+x));
            auto srcnp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcpn+x-2));
            auto dst = planar_detect_edges_sse2_core(src, srcn, srcnp, thresh_vector, ff);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp+x), dst);
        }

        *reinterpret_cast<uint16_t*>(dstp+width-2) = 0;

        srcp+=src_pitch;
        srcpn+=src_pitch;
        dstp+=dst_pitch;
    }
}

static void planar_detect_edges_c(uint8_t *dstp, const uint8_t *srcp, int dst_pitch, int src_pitch, int height, int width, int threshold, bool mask) {
    threshold = std::max(1, threshold);
    const uint8_t *srcpn = srcp+src_pitch;
    memset(dstp, 0, height * dst_pitch);

    for (int y = 0; y < height-1; y++)
    {
        for(int x = 2; x < width-2; ++x) {
            if ((std::abs(srcp[x] - srcpn[x-2]) > threshold) ||
                (std::abs(srcp[x] - srcpn[x]) > threshold)) 
            {
                dstp[x] = 255;
            } 
        }

        srcp+=src_pitch;
        srcpn+=src_pitch;
        dstp+=dst_pitch;
    }
}

static void planar_detect_edges_hiq_c(uint8_t *dstp, const uint8_t *srcp, int dst_pitch, int src_pitch, int height, int width, int threshold) {
    memset(dstp, 0, dst_pitch*2);
    dstp += dst_pitch*2;
    srcp += src_pitch*2;

    const uint8_t *srcp_saved = srcp;
    uint8_t *dstp_saved = dstp;

    for (int y = 2; y < height-2; y++) { 
        const uint8_t *srcpn = srcp + src_pitch;

        for (int x = 0; x < width; x++) {
            if (std::abs(srcp[x]-srcpn[x]) >= threshold) {
                dstp[x]=255;
            }
        }

        dstp+=dst_pitch;
        srcp+=src_pitch;
    }

    srcp = srcp_saved;
    dstp = dstp_saved;
    for (int y = 2; y < height-2; y++)
    {
        *reinterpret_cast<uint16_t*>(dstp) = 0;
        for (int x = 2; x < width-2; x++)
        {
            if (std::abs(srcp[x]-srcp[x+1])>=threshold) {
                dstp[x]=255;
            }
        }
        *reinterpret_cast<uint16_t*>(dstp+width-2) = 0;

        dstp+=dst_pitch;
        srcp+=src_pitch;
    }

    memset(dstp, 0, dst_pitch*2);
}


static void planar_detect_edges_hiq_sse2(uint8_t *dstp, const uint8_t *srcp, int dst_pitch, int src_pitch, int height, int width, int threshold) {
    memset(dstp, 0, dst_pitch*2);
    dstp += dst_pitch*2;
    srcp += src_pitch*2;
    __m128i thresh_vector = _mm_set1_epi8(threshold);
    __m128i zero = _mm_setzero_si128();
    int loop_limit = std::min(width+15, dst_pitch);
    const uint8_t *srcp_saved = srcp;
    uint8_t *dstp_saved = dstp;

    for (int y = 2; y < height-2; y++) { 
        const uint8_t *srcpn = srcp + src_pitch;

        for (int x = 0; x < loop_limit; x+=16) {
            __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x));
            __m128i srcn = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcpn+x));
            __m128i adiff = abs_diff(src, srcn);
            __m128i mask = _mm_subs_epu8(thresh_vector, adiff);
            mask = _mm_cmpeq_epi8(mask, zero);
            __m128i dst = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+x));
            dst = _mm_or_si128(mask, dst);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp+x), dst);
        }

        dstp+=dst_pitch;
        srcp+=src_pitch;
    }


    srcp = srcp_saved;
    dstp = dstp_saved;
    for (int y = 2; y < height-2; y++) {
        *reinterpret_cast<uint16_t*>(dstp) = 0;

        for (int x = 2; x < loop_limit; x+=16)
        {
            __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x));
            __m128i srcn = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+x+1));
            __m128i adiff = abs_diff(src, srcn);
            __m128i mask = _mm_subs_epu8(thresh_vector, adiff);
            mask = _mm_cmpeq_epi8(mask, zero);
            __m128i dst = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+x));
            dst = _mm_or_si128(mask, dst);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp+x), dst);
        }
        *reinterpret_cast<uint16_t*>(dstp+width-2) = 0;

        dstp+=dst_pitch;
        srcp+=src_pitch;
    }

    memset(dstp, 0, dst_pitch*2);
}

static void planar_apply_filter_c(uint8_t *dstp, const uint8_t *srcp, const uint8_t *blurp, int dst_pitch, int src_pitch, int blur_pitch, int height, int width, int strength) {
    int invstrength = 255 - strength;

    memcpy(dstp, srcp, width);
    
    srcp += src_pitch;
    dstp += dst_pitch;
    blurp += dst_pitch;

    for (int y = 1; y < height-1; y++)
    { 
        dstp[0] = srcp[0];
        for (int x = 1; x < width-1; ++x)
        {
            if (dstp[x]) {                                     
                int t = 4 * srcp[x] - 3 * blurp[x];
                if (t<0) { 
                    t = 0;
                } else if (t>255) {
                    t = 255;
                }
                dstp[x] = (strength * t + invstrength * srcp[x]) >> 8;
            }
            else {
                dstp[x] = srcp[x];
            }
        }  
        dstp[width-1] = srcp[width-1]; 

        srcp += src_pitch;
        dstp += dst_pitch;
        blurp += blur_pitch;
    }   

    memcpy(dstp, srcp, width);
}

//mask ? a : b
static __forceinline __m128i blend_si128(__m128i const &mask, __m128i const &desired, __m128i const &otherwise) {
    //return  _mm_blendv_epi8 (otherwise, desired, mask);
    auto andop = _mm_and_si128(mask , desired);
    auto andnop = _mm_andnot_si128(mask, otherwise);
    return _mm_or_si128(andop, andnop);
}

static void planar_apply_filter_sse2(uint8_t *dstp, const uint8_t *srcp, const uint8_t *blurp, int dst_pitch, int src_pitch, int blur_pitch, int height, int width, int strength) {
    memcpy(dstp, srcp, width);

    srcp += src_pitch;
    dstp += dst_pitch;
    blurp += blur_pitch;
    int loop_limit = std::min(width+7, dst_pitch);

    __m128i zero = _mm_setzero_si128();
    __m128i v255 = _mm_set1_epi16(255);
    __m128i strength_vector = _mm_set1_epi16(strength);
    __m128i invstrength_vector = _mm_set1_epi16(255 - strength);

    for (int y = 1; y < height-1; y++)
    { 
        for (int x = 0; x < loop_limit; x+=8)
        {
            __m128i src_packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x));
            __m128i blur = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(blurp+x));

            auto src = _mm_unpacklo_epi8(src_packed, zero);
            blur = _mm_unpacklo_epi8(blur, zero);

            auto t1 = _mm_slli_epi16(src, 2);
            auto t2 = _mm_subs_epu16(_mm_slli_epi16(blur, 2), blur);

            auto t = _mm_subs_epu16(t1, t2);
            t = _mm_min_epu16(t, v255);

            t1 = _mm_mullo_epi16(strength_vector, t);
            t2 = _mm_mullo_epi16(invstrength_vector, src);

            __m128i sharp = _mm_adds_epu16(t1, t2);
            sharp = _mm_srli_epi16(sharp, 8);
            sharp = _mm_packus_epi16(sharp, zero);

            __m128i mask = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(dstp+x));

            auto dst = blend_si128(mask, sharp, src_packed);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp+x), dst);
        }  

        srcp += src_pitch;
        dstp += dst_pitch;
        blurp += blur_pitch;
    }   

    memcpy(dstp, srcp, width);
}



static void rgb_blur_c(uint8_t *blurp, const uint8_t *srcp, int blur_pitch, int src_pitch, int height, int width) {
    memcpy(blurp, srcp, width);

    const uint8_t *srcpp = srcp;
    srcp += src_pitch;
    const uint8_t *srcpn = srcp + src_pitch;

    blurp += blur_pitch;

    for (int y = 1; y < height-1; ++y) {
        *reinterpret_cast<uint32_t*>(blurp) = *reinterpret_cast<const uint32_t*>(srcp);
        for (int x = 4; x < width-4; ++x) {
            uint8_t c1 = (srcpp[x-4] + srcp[x-4] + srcpn[x-4]) / 3;
            uint8_t c2 = (srcpp[x+0] + srcp[x+0] + srcpn[x+0]) / 3;
            uint8_t c3 = (srcpp[x+4] + srcp[x+4] + srcpn[x+4]) / 3;
            blurp[x] = (c1+c2+c3) / 3;
        }
        *reinterpret_cast<uint32_t*>(blurp + width - 4) = *reinterpret_cast<const uint32_t*>(srcp + width - 4);

        srcpp += src_pitch;
        srcp += src_pitch;
        srcpn += src_pitch;
        blurp += blur_pitch;
    }

    memcpy(blurp, srcp, width);
}

static void rgb_detect_edges_c(uint8_t *dstp, const uint8_t *blurp, int dst_pitch, int blur_pitch, int height, int width, int threshold) {
    const uint8_t* blurpn = blurp + blur_pitch;
    for (int y = 0; y < height - 1; y++) 
    {
        for (int x = 0; x < width - 4; x+=4)
        {
            if ((std::abs(blurp[x+0] - blurpn[x+4]) >= threshold) || 
                (std::abs(blurp[x+1] - blurpn[x+5]) >= threshold) || 
                (std::abs(blurp[x+2] - blurpn[x+6]) >= threshold) ||
                (std::abs(blurp[x+4] - blurpn[x+0]) >= threshold) || 
                (std::abs(blurp[x+5] - blurpn[x+1]) >= threshold) || 
                (std::abs(blurp[x+6] - blurpn[x+2]) >= threshold))
            {
                 *reinterpret_cast<uint32_t*>(dstp+x) = 0xffffffff;
            }
            else
            {
                *reinterpret_cast<uint32_t*>(dstp+x) = 0x0;
            }
        }
        dstp += dst_pitch;
        blurp += blur_pitch;
        blurpn += blur_pitch;
    }
}

static void rgb_detect_edges_hiq(uint8_t *dstp, const uint8_t *blurp, int dst_pitch, int blur_pitch, int height, int row_size, int threshold) {
    /* Vertical detail detection. */
    const uint8_t *blurp_saved = blurp;
    uint8_t* dstp_saved = dstp;

    const uint8_t *blurpn = blurp + blur_pitch;

    for (int y = 0; y < height - 1; y++)
    {
        for (int x = 0; x < row_size; x+=4)
        {
            if (std::abs(blurp[x+0] - blurpn[x+0]) >= threshold || 
                std::abs(blurp[x+1] - blurpn[x+1]) >= threshold || 
                std::abs(blurp[x+2] - blurpn[x+2]) >= threshold)
            {
                *reinterpret_cast<uint32_t*>(dstp+x) = 0xffffffff;
            }
        }
        dstp += dst_pitch;
        blurp += blur_pitch;
        blurpn += blur_pitch;
    }

    /* Horizontal detail detection. */
    blurp = blurp_saved;
    dstp = dstp_saved;
    for (int y = 0; y < height-1; y++)
    {
        for (int x = 0; x < row_size - 4; x+=4)
        {
            if (std::abs(blurp[x+0] - blurp[x+4]) >= threshold || 
                std::abs(blurp[x+1] - blurp[x+5]) >= threshold || 
                std::abs(blurp[x+2] - blurp[x+6]) >= threshold)
            {
                *reinterpret_cast<uint32_t*>(dstp+x) = 0xffffffff;
            }
        }

        *reinterpret_cast<uint32_t*>(dstp+row_size-4) = 0;

        dstp += dst_pitch;
        blurp += blur_pitch;
    }

    /* Fix up detail map borders. */
    memset(dstp, 0, row_size);
}

static void rgb_apply_filter(uint8_t *dstp, const uint8_t *blurp, const uint8_t *srcp, int dst_pitch, int blur_pitch, int src_pitch, int height, int width, int strength) {
    int invstrength = 255 - strength;

    memcpy(dstp, srcp, width);
    
    srcp += src_pitch;
    dstp += dst_pitch;
    blurp += blur_pitch;
    
    for (int y = 1; y < height - 1; y++)
    {
        *reinterpret_cast<uint32_t*>(dstp) = *reinterpret_cast<const uint32_t*>(srcp);

        for (int x = 4; x < width - 4; x+=4)
        {
            if (dstp[x])
            {
                int b = 4*srcp[x+0] - 3*blurp[x+0];
                int g = 4*srcp[x+1] - 3*blurp[x+1];
                int r = 4*srcp[x+2] - 3*blurp[x+2];

                if (b < 0) b = 0;
                if (g < 0) g = 0;
                if (r < 0) r = 0;
                int max = b;
                if (g > max) max = g;
                if (r > max) max = r;
                if (max > 255)
                {
                    b = (b * 255) / max;
                    g = (g * 255) / max;
                    r = (r * 255) / max;
                }
                dstp[x+0] = (strength * b + invstrength * srcp[x+0]) >> 8;
                dstp[x+1] = (strength * g + invstrength * srcp[x+1]) >> 8;
                dstp[x+2] = (strength * r + invstrength * srcp[x+2]) >> 8;
                dstp[x+3] = srcp[x+3];
            }
            else
            {
                *reinterpret_cast<uint32_t*>(dstp+x) = *reinterpret_cast<const uint32_t*>(srcp+x); 
            }
        }

        *reinterpret_cast<uint32_t*>(dstp+width-4) = *reinterpret_cast<const uint32_t*>(srcp+width-4);

        srcp += src_pitch;
        dstp += dst_pitch;
        blurp += blur_pitch;
    }

    memcpy(dstp, srcp, width);
}

class MSharpen : public GenericVideoFilter {
public:
    MSharpen(PClip child, int threshold, int strength, bool highq, bool mask, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);
    ~MSharpen() {
        _aligned_free(blur_buffer);
    }

private:
    int threshold_;
    int strength_;
    bool highq_;
    bool show_mask_;

    uint8_t* blur_buffer;
    size_t blur_pitch;
};


MSharpen::MSharpen(PClip child, int threshold, int strength, bool highq, bool mask, IScriptEnvironment* env)
: GenericVideoFilter(child), threshold_(threshold), strength_(strength),
highq_(highq), show_mask_(mask), blur_buffer(nullptr)
{
    if (!(vi.IsPlanar() || vi.IsRGB32())) {
        env->ThrowError("MSharpen: YUY2, RGB32 or planar (YV12 for instance) color space required");
    }
    if (strength < 0 || strength > 255) {
        env->ThrowError("MSharpen: strength out of range (0-255)");
    }
    int bytes_width = vi.IsRGB32() ? vi.width * 4 : vi.width;

    blur_pitch = (bytes_width + 15) / 16 * 16;
    blur_buffer = reinterpret_cast<uint8_t*>(_aligned_malloc(blur_pitch * vi.height, 16));
    if (!blur_buffer) {
        env->ThrowError("MSharpen: malloc failure");
    }
}

PVideoFrame __stdcall MSharpen::GetFrame(int n, IScriptEnvironment *env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    if (vi.IsPlanar())
    {
        int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };
        int limit = vi.IsY8() ? 1 : 3;
        for (int i = 0; i < limit; ++i)
        {
            int plane = planes[i];

            uint8_t *dstp = dst->GetWritePtr(plane);
            const uint8_t *srcp = src->GetReadPtr(plane);
            int src_pitch = src->GetPitch(plane);
            int dst_pitch = dst->GetPitch(plane);
            int width = src->GetRowSize(plane);
            int height = src->GetHeight(plane);

            if (env->GetCPUFlags() & CPUF_SSE2) {
                planar_blur_sse2(blur_buffer, srcp, blur_pitch, src_pitch, height, width);

                if (is_ptr_aligned(srcp, 16)) {
                    planar_detect_edges_sse2(dstp, blur_buffer, dst_pitch, blur_pitch, height, width, threshold_, show_mask_);
                    if (highq_) {
                        planar_detect_edges_hiq_sse2(dstp, blur_buffer, dst_pitch, blur_pitch, height, width, threshold_);
                    }
                } else {
                    planar_detect_edges_c(dstp, blur_buffer, dst_pitch, blur_pitch, height, width, threshold_, show_mask_);
                    if (highq_) {
                        planar_detect_edges_hiq_c(dstp, blur_buffer, dst_pitch, blur_pitch, height, width, threshold_);
                    }
                }

                if (!show_mask_) {
                    planar_apply_filter_sse2(dstp, srcp, blur_buffer, dst_pitch, src_pitch, blur_pitch, height, width, strength_);
                }
            } else {
                planar_blur_c(blur_buffer, srcp, blur_pitch, src_pitch, height, width);
                planar_detect_edges_c(dstp, blur_buffer, dst_pitch, blur_pitch, height, width, threshold_, show_mask_);

                if (highq_) {
                    planar_detect_edges_hiq_c(dstp, blur_buffer, dst_pitch, blur_pitch, height, width, threshold_);
                }

                if (!show_mask_) {
                    planar_apply_filter_c(dstp, srcp, blur_buffer, dst_pitch, src_pitch, blur_pitch, height, width, strength_);
                }
            }
        }
        return dst;
    }

	const uint8_t *srcp = src->GetReadPtr();
	int src_pitch = src->GetPitch();

	uint8_t *dstp = dst->GetWritePtr();
    int dst_pitch = dst->GetPitch();

	int width = src->GetRowSize();
	int height = src->GetHeight();

    rgb_blur_c(blur_buffer, srcp, blur_pitch, src_pitch, height, width);

    rgb_detect_edges_c(dstp, blur_buffer, dst_pitch, blur_pitch, height, width, threshold_);

    if (highq_) {
        rgb_detect_edges_hiq(dstp, blur_buffer, dst_pitch, blur_pitch, height, width, threshold_);
    }

    if (show_mask_) {
        return dst;
    }

    rgb_apply_filter(dstp, blur_buffer, srcp, dst_pitch, blur_pitch, src_pitch, height, width, strength_);
    
    return dst;
}


AVSValue __cdecl Create_MSharpen(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, THRESH, STRENGTH, HIGHQ, MASK };
    return new MSharpen(args[CLIP].AsClip(), args[THRESH].AsInt(15), args[STRENGTH].AsInt(100), args[HIGHQ].AsBool(true), args[MASK].AsBool(false), env);
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;

    env->AddFunction("MSharpen", "c[threshold]i[strength]i[highq]b[mask]b", Create_MSharpen, 0);
    return "Doushimashita?";
}