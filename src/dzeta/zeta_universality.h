#pragma once

#include "zeta_rhythm.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

namespace dzeta {

using ZetaComplex = std::complex<long double>;

struct ZetaPoint {
    long double real = 0.0L;
    long double imag = 0.0L;
};

struct ZetaUniversalityReport {
    long double functional_residual = 0.0L;
    long double universality_score = 0.0L;
    ZetaComplex xi_s{0.0L, 0.0L};
    ZetaComplex xi_reflected{0.0L, 0.0L};
};

inline ZetaComplex complex_gamma(ZetaComplex z) {
    return std::exp(lanczos_log_gamma(z));
}

inline ZetaComplex dirichlet_eta(ZetaComplex s, std::size_t terms) {
    ZetaComplex sum{0.0L, 0.0L};
    for (std::size_t n = 1; n <= terms; ++n) {
        const long double sign = (n % 2U == 0U) ? -1.0L : 1.0L;
        sum += sign * std::exp(-s * std::log(static_cast<long double>(n)));
    }
    return sum;
}

inline ZetaComplex zeta_numeric(ZetaComplex s, std::size_t terms = 128) {
    if (std::abs(s - ZetaComplex{1.0L, 0.0L}) < 1.0e-12L) {
        return {std::numeric_limits<long double>::infinity(), 0.0L};
    }
    if (s.real() > 0.0L) {
        const auto eta = dirichlet_eta(s, terms);
        const auto denominator = ZetaComplex{1.0L, 0.0L} -
                                 std::exp((ZetaComplex{1.0L, 0.0L} - s) * std::log(2.0L));
        return eta / denominator;
    }
    const auto one_minus = ZetaComplex{1.0L, 0.0L} - s;
    const auto factor = std::exp(s * std::log(2.0L)) *
                        std::exp((s - ZetaComplex{1.0L, 0.0L}) * std::log(DZETA_PI)) *
                        std::sin(0.5L * DZETA_PI * s) *
                        complex_gamma(one_minus);
    return factor * zeta_numeric(one_minus, terms);
}

inline ZetaComplex completed_zeta_xi(ZetaComplex s, std::size_t terms = 128) {
    const auto half = ZetaComplex{0.5L, 0.0L};
    return half * s * (s - ZetaComplex{1.0L, 0.0L}) *
           std::exp((-0.5L * s) * std::log(DZETA_PI)) *
           complex_gamma(0.5L * s) *
           zeta_numeric(s, terms);
}

inline ZetaComplex hardy_z(long double t, std::size_t terms = 128) {
    const long double theta = riemann_siegel_theta(t);
    const std::size_t limit = std::max<std::size_t>(
        1,
        std::min<std::size_t>(terms, static_cast<std::size_t>(std::sqrt(t / DZETA_TWO_PI))));
    long double sum = 0.0L;
    for (std::size_t n = 1; n <= limit; ++n) {
        const long double ln = std::log(static_cast<long double>(n));
        sum += std::cos(theta - t * ln) / std::sqrt(static_cast<long double>(n));
    }
    return {2.0L * sum, 0.0L};
}

inline ZetaUniversalityReport zeta_functional_equation_report(ZetaPoint point, std::size_t samples) {
    ZetaUniversalityReport report;
    const ZetaComplex s{point.real, point.imag};
    report.xi_s = completed_zeta_xi(s, samples);
    report.xi_reflected = completed_zeta_xi(ZetaComplex{1.0L, 0.0L} - s, samples);
    const auto denom = 1.0L + std::max(std::abs(report.xi_s), std::abs(report.xi_reflected));
    report.functional_residual = std::abs(report.xi_s - report.xi_reflected) / denom;

    long double best = 1.0e100L;
    const ZetaComplex target = zeta_numeric(s, samples);
    for (std::size_t i = 1; i <= samples; ++i) {
        const long double tau = static_cast<long double>(i) * 0.125L;
        const auto shifted = zeta_numeric(s + ZetaComplex{0.0L, tau}, samples);
        best = std::min(best, static_cast<long double>(std::abs(shifted - target)));
    }
    report.universality_score = std::clamp(1.0L / (1.0L + best), 0.0L, 1.0L);
    return report;
}

} // namespace dzeta
