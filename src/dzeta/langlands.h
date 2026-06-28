#pragma once

#include "handle.h"
#include "sat.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dzeta {

struct LanglandsLocalFactor {
    std::uint32_t prime = 0;
    long double frobenius_trace = 0.0L;
    long double local_factor_norm = 0.0L;
    long double conductor_penalty = 0.0L;
};

struct LanglandsSignature {
    std::vector<LanglandsLocalFactor> local_factors;
    long double global_norm = 0.0L;
    long double conductor = 0.0L;
    long double euler_product_abs = 1.0L;
    std::uint64_t representation_hash = 0;
};

inline LanglandsSignature langlands_signature(const std::vector<Handle>& handles) {
    LanglandsSignature signature;
    signature.local_factors.reserve(handles.size());
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (const auto& handle : handles) {
        LanglandsLocalFactor factor;
        factor.prime = handle.prime;
        factor.frobenius_trace = std::cos(handle.theta) + handle.activation * std::sin(handle.phase);
        factor.conductor_penalty = std::log1p(static_cast<long double>(handle.prime)) *
                                   (1.0L + handle.resistance) / (1.0L + handle.activation);
        factor.local_factor_norm = std::log1p(std::abs(factor.frobenius_trace)) /
                                   (1.0L + factor.conductor_penalty);
        const long double local_euler = 1.0L /
                                        std::max(1.0e-12L,
                                                 std::abs(1.0L -
                                                          factor.frobenius_trace /
                                                              std::sqrt(static_cast<long double>(handle.prime)) +
                                                          1.0L / static_cast<long double>(handle.prime)));
        signature.global_norm += factor.local_factor_norm;
        signature.conductor += factor.conductor_penalty;
        signature.euler_product_abs *= std::clamp(local_euler, 0.25L, 4.0L);
        hash ^= stable_hash(std::to_string(handle.prime) + ":" +
                            std::to_string(static_cast<double>(factor.frobenius_trace)));
        hash = splitmix64(hash);
        signature.local_factors.push_back(factor);
    }
    if (!handles.empty()) {
        signature.global_norm /= static_cast<long double>(handles.size());
        signature.conductor /= static_cast<long double>(handles.size());
        signature.euler_product_abs = std::pow(signature.euler_product_abs,
                                               1.0L / static_cast<long double>(handles.size()));
    }
    signature.representation_hash = hash;
    return signature;
}

inline long double langlands_compatibility(const LanglandsSignature& left,
                                           const LanglandsSignature& right) {
    const long double norm_gap = std::abs(left.global_norm - right.global_norm);
    const long double conductor_gap = std::abs(left.conductor - right.conductor) /
                                      (1.0L + std::max(std::abs(left.conductor), std::abs(right.conductor)));
    const std::size_t count = std::min(left.local_factors.size(), right.local_factors.size());
    long double local_gap = 0.0L;
    for (std::size_t i = 0; i < count; ++i) {
        local_gap += std::abs(left.local_factors[i].local_factor_norm -
                              right.local_factors[i].local_factor_norm);
    }
    if (count != 0) {
        local_gap /= static_cast<long double>(count);
    }
    return std::clamp(1.0L / (1.0L + norm_gap + conductor_gap + local_gap), 0.0L, 1.0L);
}

} // namespace dzeta
