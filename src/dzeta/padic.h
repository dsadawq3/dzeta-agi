#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace dzeta {

inline std::uint32_t padic_valuation(std::uint64_t base_prime, std::uint64_t value) {
    if (base_prime < 2) {
        return 0;
    }
    if (value == 0) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    std::uint32_t power = 0;
    while (value % base_prime == 0) {
        value /= base_prime;
        ++power;
    }
    return power;
}

inline long double padic_norm(std::uint64_t base_prime, std::uint64_t value) {
    if (value == 0) {
        return 0.0L;
    }
    const auto valuation = padic_valuation(base_prime, value);
    return std::pow(static_cast<long double>(base_prime), -static_cast<long double>(valuation));
}

inline long double padic_distance(std::uint64_t base_prime, std::int64_t left, std::int64_t right) {
    const auto delta = left >= right
                           ? static_cast<std::uint64_t>(left - right)
                           : static_cast<std::uint64_t>(right - left);
    return padic_norm(base_prime, delta);
}

struct PadicBall {
    std::uint64_t base_prime = 2;
    std::int64_t center = 0;
    std::uint32_t radius_power = 0;
    std::uint64_t modulus = 1;
    std::int64_t residue = 0;
    long double radius = 1.0L;
};

inline std::uint64_t checked_ipow(std::uint64_t base, std::uint32_t exponent) {
    std::uint64_t value = 1;
    for (std::uint32_t i = 0; i < exponent; ++i) {
        if (base != 0 && value > std::numeric_limits<std::uint64_t>::max() / base) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        value *= base;
    }
    return value;
}

inline std::int64_t positive_mod(std::int64_t value, std::uint64_t modulus) {
    if (modulus == 0) {
        return value;
    }
    const auto signed_modulus = static_cast<std::int64_t>(std::min<std::uint64_t>(
        modulus, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
    const auto residue = value % signed_modulus;
    return residue < 0 ? residue + signed_modulus : residue;
}

inline PadicBall minimal_padic_ball(std::uint64_t base_prime, std::int64_t left, std::int64_t right) {
    const auto delta = left >= right
                           ? static_cast<std::uint64_t>(left - right)
                           : static_cast<std::uint64_t>(right - left);
    const auto radius_power = delta == 0 ? 32U : padic_valuation(base_prime, delta);
    const auto modulus = checked_ipow(base_prime, radius_power);
    PadicBall ball;
    ball.base_prime = base_prime;
    ball.center = left;
    ball.radius_power = radius_power;
    ball.modulus = modulus;
    ball.residue = positive_mod(left, modulus);
    ball.radius = padic_norm(base_prime, delta);
    return ball;
}

inline bool in_padic_ball(const PadicBall& ball, std::int64_t value) {
    if (ball.radius_power == 0) {
        return true;
    }
    return positive_mod(value, ball.modulus) == ball.residue;
}

inline std::vector<std::uint32_t> padic_cluster_signature(std::uint64_t value,
                                                          std::initializer_list<std::uint64_t> bases) {
    std::vector<std::uint32_t> signature;
    signature.reserve(bases.size());
    for (const auto base : bases) {
        signature.push_back(padic_valuation(base, value));
    }
    return signature;
}

inline std::string padic_hierarchy_key(std::uint64_t value,
                                       std::initializer_list<std::uint64_t> bases,
                                       std::uint32_t depth) {
    std::ostringstream key;
    bool first = true;
    for (const auto base : bases) {
        if (!first) {
            key << "/";
        }
        first = false;
        const auto modulus = checked_ipow(base, depth);
        key << base << ":" << positive_mod(static_cast<std::int64_t>(value), modulus);
    }
    return key.str();
}

struct AdelicProfile {
    std::uint64_t value = 0;
    std::vector<std::uint64_t> bases;
    std::vector<std::uint32_t> valuations;
    std::vector<std::int64_t> residues;
    std::string hierarchy_key;
};

inline AdelicProfile adelic_profile(std::uint64_t value,
                                    std::initializer_list<std::uint64_t> bases,
                                    std::uint32_t depth) {
    AdelicProfile profile;
    profile.value = value;
    profile.bases.assign(bases.begin(), bases.end());
    profile.valuations.reserve(profile.bases.size());
    profile.residues.reserve(profile.bases.size());
    for (const auto base : profile.bases) {
        profile.valuations.push_back(padic_valuation(base, value));
        const auto modulus = checked_ipow(base, depth);
        profile.residues.push_back(positive_mod(static_cast<std::int64_t>(value), modulus));
    }
    profile.hierarchy_key = padic_hierarchy_key(value, bases, depth);
    return profile;
}

inline long double adelic_ultrametric_distance(std::uint64_t left,
                                               std::uint64_t right,
                                               std::initializer_list<std::uint64_t> bases,
                                               std::uint32_t depth) {
    if (left == right) {
        return 0.0L;
    }
    const auto delta = left > right ? left - right : right - left;
    long double distance = 0.0L;
    for (const auto base : bases) {
        const auto valuation = std::min(padic_valuation(base, delta), depth);
        const long double component = std::pow(static_cast<long double>(base),
                                               -static_cast<long double>(valuation));
        distance = std::max(distance, component);
    }
    return distance;
}

inline long double symmetric_prime_distance(std::uint32_t left_prime, std::uint32_t right_prime) {
    if (left_prime == right_prime) {
        return 0.0L;
    }
    const auto gap = left_prime > right_prime ? left_prime - right_prime : right_prime - left_prime;
    const long double two_adic = padic_distance(2, left_prime, right_prime);
    const long double scale = 1.0L + std::log1pl(static_cast<long double>(gap)) /
                                       std::log1pl(static_cast<long double>(std::max(left_prime, right_prime)));
    return two_adic * scale;
}

} // namespace dzeta
