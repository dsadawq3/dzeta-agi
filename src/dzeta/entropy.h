#pragma once

#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace dzeta {

enum class EntropyMode {
    RequireRdrand,
    AllowFallback,
};

inline bool runtime_rdrand_supported() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }
    return (ecx & (1U << 30U)) != 0U;
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int cpu_info[4] = {};
    __cpuid(cpu_info, 1);
    return (cpu_info[2] & (1 << 30)) != 0;
#else
    return false;
#endif
}

inline bool rdrand64_runtime(std::uint64_t& value) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    unsigned char ok = 0;
    unsigned long long out = 0;
    __asm__ __volatile__("rdrand %0; setc %1" : "=r"(out), "=qm"(ok));
    value = static_cast<std::uint64_t>(out);
    return ok != 0;
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__i386__)
    unsigned char ok_low = 0;
    unsigned char ok_high = 0;
    unsigned int low = 0;
    unsigned int high = 0;
    __asm__ __volatile__("rdrand %0; setc %1" : "=r"(low), "=qm"(ok_low));
    __asm__ __volatile__("rdrand %0; setc %1" : "=r"(high), "=qm"(ok_high));
    value = (static_cast<std::uint64_t>(high) << 32U) | static_cast<std::uint64_t>(low);
    return ok_low != 0 && ok_high != 0;
#elif defined(_MSC_VER) && defined(_M_X64)
    unsigned __int64 out = 0;
    const int ok = _rdrand64_step(&out);
    value = static_cast<std::uint64_t>(out);
    return ok == 1;
#elif defined(_MSC_VER) && defined(_M_IX86)
    unsigned int low = 0;
    unsigned int high = 0;
    const int ok_low = _rdrand32_step(&low);
    const int ok_high = _rdrand32_step(&high);
    value = (static_cast<std::uint64_t>(high) << 32U) | static_cast<std::uint64_t>(low);
    return ok_low == 1 && ok_high == 1;
#else
    value = 0;
    return false;
#endif
}

inline std::uint64_t nondeterministic_fallback_seed(std::uint64_t seed) {
    std::random_device rd;
    std::uint64_t mixed = seed ^ (static_cast<std::uint64_t>(rd()) << 32U) ^ static_cast<std::uint64_t>(rd());
    mixed ^= 0x9e3779b97f4a7c15ULL + (mixed << 6U) + (mixed >> 2U);
    return mixed;
}

class HardwareEntropy {
public:
    explicit HardwareEntropy(std::uint64_t seed = std::random_device{}(),
                             EntropyMode mode = EntropyMode::RequireRdrand)
        : rdrand_available_(runtime_rdrand_supported()),
          mode_(mode),
          fallback_(nondeterministic_fallback_seed(seed)),
          distribution_(0.0L, 1.0L) {
        if (mode_ == EntropyMode::RequireRdrand && !rdrand_available_) {
            throw std::runtime_error("RDRAND is required, but CPUID does not report RDRAND support");
        }
    }

    long double next01() {
        if (rdrand_available_) {
            for (int attempt = 0; attempt < 10; ++attempt) {
                std::uint64_t value = 0;
                if (rdrand64_runtime(value)) {
                    return static_cast<long double>(value) /
                           static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
                }
            }
        }
        if (mode_ == EntropyMode::RequireRdrand) {
            throw std::runtime_error("RDRAND failed repeatedly during entropy sampling");
        }
        return distribution_(fallback_);
    }

    long double signed_unit() {
        return next01() * 2.0L - 1.0L;
    }

    std::string_view provider() const {
        if (rdrand_available_) {
            return "rdrand-runtime";
        }
        return "std::random_device/mt19937_64 fallback";
    }

private:
    bool rdrand_available_ = false;
    EntropyMode mode_ = EntropyMode::RequireRdrand;
    std::mt19937_64 fallback_;
    std::uniform_real_distribution<long double> distribution_;
};

} // namespace dzeta
