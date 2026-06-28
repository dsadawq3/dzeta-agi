#pragma once

#include "zeta_zeros.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>

namespace dzeta {

inline constexpr long double DZETA_PI = 3.141592653589793238462643383279502884L;
inline constexpr long double DZETA_TWO_PI = 2.0L * DZETA_PI;

inline long double wrap_phase(long double value) {
    long double wrapped = std::fmod(value + DZETA_PI, DZETA_TWO_PI);
    if (wrapped < 0.0L) {
        wrapped += DZETA_TWO_PI;
    }
    return wrapped - DZETA_PI;
}

inline long double phase_distance(long double left, long double right) {
    return std::abs(wrap_phase(left - right));
}

inline long double riemann_siegel_theta_asymptotic(long double t) {
    if (t <= 0.0L) {
        return 0.0L;
    }
    const long double t2 = t * t;
    const long double t3 = t2 * t;
    const long double t5 = t3 * t2;
    return 0.5L * t * std::log(t / (2.0L * DZETA_PI)) - 0.5L * t - DZETA_PI / 8.0L +
           1.0L / (48.0L * t) + 7.0L / (5760.0L * t3) + 31.0L / (80640.0L * t5);
}

inline std::complex<long double> lanczos_log_gamma(std::complex<long double> z) {
    static constexpr long double g = 7.0L;
    static constexpr long double coefficients[] = {
        0.99999999999980993227684700473478L,
        676.52036812188509856700919044402L,
        -1259.1392167224028704715607875528L,
        771.3234287776530788486528258894L,
        -176.61502916214059906584551354L,
        12.507343278686904814458936853L,
        -0.13857109526572011689554707L,
        0.000009984369578019570859563L,
        0.000000150563273514931155834L,
    };

    z -= std::complex<long double>{1.0L, 0.0L};
    std::complex<long double> x{coefficients[0], 0.0L};
    for (std::size_t i = 1; i < std::size(coefficients); ++i) {
        x += coefficients[i] / (z + static_cast<long double>(i));
    }

    const auto t = z + std::complex<long double>{g + 0.5L, 0.0L};
    return std::complex<long double>{0.5L * std::log(DZETA_TWO_PI), 0.0L} +
           (z + std::complex<long double>{0.5L, 0.0L}) * std::log(t) - t + std::log(x);
}

inline long double riemann_siegel_theta_from_log_gamma(long double t) {
    if (t <= 0.0L) {
        return 0.0L;
    }

    const std::complex<long double> z{0.25L, 0.5L * t};
    long double theta = std::imag(lanczos_log_gamma(z)) - 0.5L * t * std::log(DZETA_PI);

    const long double reference = riemann_siegel_theta_asymptotic(t);
    const long double turns = std::round((reference - theta) / DZETA_TWO_PI);
    theta += turns * DZETA_TWO_PI;
    return theta;
}

inline std::complex<long double> zeta_rhythm(std::uint32_t prime,
                                             long double tick_time,
                                             std::size_t zeros_to_use = zeta_zero_count()) {
    const std::size_t limit = std::min(zeros_to_use, zeta_zero_count());
    std::complex<long double> sum{0.0L, 0.0L};
    for (std::size_t index = 0; index < limit; ++index) {
        const long double angle = zeta_zero(index) * static_cast<long double>(prime) * tick_time;
        sum += std::complex<long double>{std::cos(angle), std::sin(angle)};
    }
    return sum;
}

inline long double zeta_phase(std::uint32_t prime,
                              long double tick_time,
                              std::size_t zeros_to_use = zeta_zero_count()) {
    const auto rhythm = zeta_rhythm(prime, tick_time, zeros_to_use);
    return std::atan2(rhythm.imag(), rhythm.real());
}

inline bool phase_coherent(std::uint32_t left_prime,
                           std::uint32_t right_prime,
                           long double tick_time,
                           long double epsilon,
                           std::size_t zeros_to_use = zeta_zero_count()) {
    return phase_distance(zeta_phase(left_prime, tick_time, zeros_to_use),
                          zeta_phase(right_prime, tick_time, zeros_to_use)) < epsilon;
}

inline long double spectral_energy(std::uint32_t prime,
                                   std::size_t zeros_to_use = zeta_zero_count()) {
    const std::size_t limit = std::min(zeros_to_use, zeta_zero_count());
    long double energy = 0.0L;
    for (std::size_t index = 0; index < limit; ++index) {
        const long double value = std::cos(zeta_zero(index) * static_cast<long double>(prime));
        energy += value * value;
    }
    return energy;
}

inline long double riemann_siegel_theta(long double t);

struct ZeroSpacingStatistics {
    std::size_t start_index = 0;
    std::size_t count = 0;
    long double mean_gap = 0.0L;
    long double variance_gap = 0.0L;
    long double normalized_mean_gap = 0.0L;
    long double min_gap = 0.0L;
    long double max_gap = 0.0L;
};

inline long double zero_gap(std::size_t index) {
    if (index + 1 >= zeta_zero_count()) {
        return 0.0L;
    }
    return zeta_zero(index + 1) - zeta_zero(index);
}

inline long double normalized_zero_gap(std::size_t index) {
    const auto gap = zero_gap(index);
    if (gap <= 0.0L || index + 1 >= zeta_zero_count()) {
        return 0.0L;
    }
    const long double midpoint = 0.5L * (zeta_zero(index) + zeta_zero(index + 1));
    return gap * std::log(midpoint / DZETA_TWO_PI) / DZETA_TWO_PI;
}

inline ZeroSpacingStatistics zero_spacing_statistics(std::size_t start_index, std::size_t zero_count) {
    ZeroSpacingStatistics stats;
    stats.start_index = start_index;
    if (zero_count < 2 || start_index + 1 >= zeta_zero_count()) {
        return stats;
    }

    const std::size_t end = std::min(zeta_zero_count(), start_index + zero_count);
    stats.count = end > start_index ? end - start_index - 1 : 0;
    if (stats.count == 0) {
        return stats;
    }

    long double sum = 0.0L;
    long double normalized_sum = 0.0L;
    stats.min_gap = std::numeric_limits<long double>::max();
    stats.max_gap = 0.0L;
    for (std::size_t index = start_index; index + 1 < end; ++index) {
        const auto gap = zero_gap(index);
        sum += gap;
        normalized_sum += normalized_zero_gap(index);
        stats.min_gap = std::min(stats.min_gap, gap);
        stats.max_gap = std::max(stats.max_gap, gap);
    }
    stats.mean_gap = sum / static_cast<long double>(stats.count);
    stats.normalized_mean_gap = normalized_sum / static_cast<long double>(stats.count);

    long double variance = 0.0L;
    for (std::size_t index = start_index; index + 1 < end; ++index) {
        const auto delta = zero_gap(index) - stats.mean_gap;
        variance += delta * delta;
    }
    stats.variance_gap = variance / static_cast<long double>(stats.count);
    return stats;
}

inline long double gram_phase_residual(long double t) {
    const long double theta = riemann_siegel_theta(t);
    return theta - std::round(theta / DZETA_PI) * DZETA_PI;
}

inline long double riemann_siegel_theta(long double t) {
    return riemann_siegel_theta_from_log_gamma(t);
}

inline long double riemann_siegel_theta(std::uint32_t prime) {
    return riemann_siegel_theta(static_cast<long double>(prime));
}

} // namespace dzeta
