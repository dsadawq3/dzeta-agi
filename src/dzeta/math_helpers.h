#pragma once
// ── math_helpers — общие inline-хелперы без математических изменений ──
// stable_hash / splitmix64, normalize, cosine. Формулы — копии из sat.h / token_field.h.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace dzeta {

// ── Hash helpers (stable, детерминированные) ──────────────────────────
inline std::uint64_t stable_hash(std::string_view text) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

// ── Normalize helpers (isfinite-guard, как в token_field.h) ───────────
#ifndef DZETA_MATH_REAL
#define DZETA_MATH_REAL double
#endif

template <typename Cx>
inline void normalize_complex_generic(std::vector<Cx>& values) {
    using Real = typename Cx::value_type;
    Real norm = 0;
    for (auto v : values) {
        if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) return;
        norm += v.real() * v.real() + v.imag() * v.imag();
    }
    if (!std::isfinite(norm) || norm <= Real(values.size()) * std::numeric_limits<Real>::epsilon() * Real(10)) return;
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm <= Real(values.size()) * std::numeric_limits<Real>::epsilon() * Real(10)) return;
    for (auto& v : values) v /= norm;
}

template <typename Real>
inline void normalize_real_generic(std::vector<Real>& values) {
    Real norm = 0;
    for (auto v : values) {
        if (!std::isfinite(v)) return;
        norm += v * v;
    }
    if (!std::isfinite(norm) || norm <= Real(values.size()) * std::numeric_limits<Real>::epsilon() * Real(10)) return;
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm <= Real(values.size()) * std::numeric_limits<Real>::epsilon() * Real(10)) return;
    for (auto& v : values) v /= norm;
}

// ── Cosine helpers ────────────────────────────────────────────────────
template <typename Real>
inline Real cosine_generic(const std::vector<Real>& left, const std::vector<Real>& right) {
    const std::size_t count = std::min(left.size(), right.size());
    if (count == 0) return Real(0);
    Real dot = 0, ln = 0, rn = 0;
    for (std::size_t i = 0; i < count; ++i) {
        dot += left[i] * right[i];
        ln += left[i] * left[i];
        rn += right[i] * right[i];
    }
    if (ln <= Real(count) * std::numeric_limits<Real>::epsilon() * Real(10) || rn <= Real(count) * std::numeric_limits<Real>::epsilon() * Real(10)) return Real(0);
    return std::clamp(dot / std::sqrt(ln * rn), Real(-1), Real(1));
}

template <typename Real>
inline Real normalized_cosine_generic(const std::vector<Real>& left, const std::vector<Real>& right) {
    const std::size_t count = std::min(left.size(), right.size());
    if (count == 0) return Real(0);
    Real dot = 0;
    for (std::size_t i = 0; i < count; ++i) dot += left[i] * right[i];
    return std::clamp(dot, Real(-1), Real(1));
}

} // namespace dzeta
