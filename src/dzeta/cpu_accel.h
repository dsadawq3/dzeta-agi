#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace dzeta {

enum class CpuDispatchMode {
    Scalar,
    Avx2,
    Avx512,
    Auto,
};

struct CpuFeatures {
    bool avx2 = false;
    bool avx512f = false;
    std::string dispatch_name = "scalar";
};

inline CpuFeatures detect_cpu_features() {
    CpuFeatures features;
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) != 0) {
        features.avx2 = (ebx & (1U << 5U)) != 0U;
        features.avx512f = (ebx & (1U << 16U)) != 0U;
    }
#endif
    if (features.avx512f) {
        features.dispatch_name = "avx512-runtime";
    } else if (features.avx2) {
        features.dispatch_name = "avx2-runtime";
    }
    return features;
}

inline CpuDispatchMode parse_cpu_dispatch(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "scalar") {
        return CpuDispatchMode::Scalar;
    }
    if (value == "avx2") {
        return CpuDispatchMode::Avx2;
    }
    if (value == "avx512") {
        return CpuDispatchMode::Avx512;
    }
    return CpuDispatchMode::Auto;
}

inline double squared_l2_scalar(const double* left, const double* right, std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double delta = left[i] - right[i];
        sum += delta * delta;
    }
    return sum;
}

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
__attribute__((target("avx2")))
inline double squared_l2_avx2_kernel(const double* left, const double* right, std::size_t count) {
    std::size_t i = 0;
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= count; i += 4) {
        const __m256d l = _mm256_loadu_pd(left + i);
        const __m256d r = _mm256_loadu_pd(right + i);
        const __m256d d = _mm256_sub_pd(l, r);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(d, d));
    }
    alignas(32) double lanes[4] = {};
    _mm256_store_pd(lanes, acc);
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < count; ++i) {
        const double delta = left[i] - right[i];
        sum += delta * delta;
    }
    return sum;
}
#endif

inline double squared_l2_auto(const double* left,
                              const double* right,
                              std::size_t count,
                              CpuDispatchMode mode = CpuDispatchMode::Auto) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    const auto features = detect_cpu_features();
    if ((mode == CpuDispatchMode::Auto || mode == CpuDispatchMode::Avx2 || mode == CpuDispatchMode::Avx512) &&
        features.avx2) {
        return squared_l2_avx2_kernel(left, right, count);
    }
#else
    (void)mode;
#endif
    return squared_l2_scalar(left, right, count);
}

#if defined(__GNUC__) || defined(__clang__)
#define DZETA_LIKELY(x) __builtin_expect(!!(x), 1)
#define DZETA_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define DZETA_PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 1)
#else
#define DZETA_LIKELY(x) (x)
#define DZETA_UNLIKELY(x) (x)
#define DZETA_PREFETCH_READ(addr) ((void)0)
#endif

} // namespace dzeta
