#pragma once

#include "sat.h"
#include "zeta_rhythm.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dzeta {

struct SleReport {
    long double kappa = 0.0L;
    long double driving_variance = 0.0L;
    long double fractal_dimension = 1.0L;
    long double trace_radius = 0.0L;
};

inline long double deterministic_normal(std::uint64_t& seed) {
    const long double u1 = (static_cast<long double>(splitmix64(seed) % 1'000'000ULL) + 1.0L) / 1'000'001.0L;
    const long double u2 = (static_cast<long double>(splitmix64(seed) % 1'000'000ULL) + 1.0L) / 1'000'001.0L;
    return std::sqrt(-2.0L * std::log(u1)) * std::cos(DZETA_TWO_PI * u2);
}

inline SleReport estimate_sle(const std::vector<long double>& phases,
                              std::size_t steps,
                              std::uint64_t seed) {
    SleReport report;
    if (phases.size() < 2 || steps == 0) {
        return report;
    }

    std::vector<long double> increments;
    increments.reserve(phases.size() - 1U);
    for (std::size_t i = 1; i < phases.size(); ++i) {
        increments.push_back(wrap_phase(phases[i] - phases[i - 1]));
    }
    long double mean = 0.0L;
    for (auto value : increments) {
        mean += value;
    }
    mean /= static_cast<long double>(increments.size());
    long double variance = 0.0L;
    for (auto value : increments) {
        const auto delta = value - mean;
        variance += delta * delta;
    }
    variance /= static_cast<long double>(increments.size());
    report.driving_variance = variance;
    report.kappa = std::clamp(4.0L * variance + 0.25L, 0.05L, 8.0L);
    report.fractal_dimension = 1.0L + std::min(report.kappa, 8.0L) / 8.0L;

    std::complex<long double> g{0.0L, 1.0L};
    long double w = 0.0L;
    const long double dt = 1.0L / static_cast<long double>(steps);
    for (std::size_t i = 0; i < steps; ++i) {
        w += std::sqrt(report.kappa * dt) * deterministic_normal(seed);
        const auto denom = g - std::complex<long double>{w, 0.0L};
        if (std::abs(denom) > 1.0e-12L) {
            g += (2.0L * dt) / denom;
        }
        report.trace_radius = std::max(report.trace_radius, static_cast<long double>(std::abs(g)));
    }
    return report;
}

} // namespace dzeta
