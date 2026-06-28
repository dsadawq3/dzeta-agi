#pragma once

#include "handle.h"
#include "primes.h"
#include "sat.h"
#include "zeta_rhythm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct FieldChart {
    std::vector<std::size_t> indices;
    std::uint32_t anchor_prime = 2;
    long double mean_phase = 0.0L;
    long double mean_activation = 0.0L;
    long double locality = 0.0L;
};

struct FieldState {
    std::vector<std::uint32_t> primes;
    std::vector<long double> phases;
    std::vector<long double> activations;
    std::vector<long double> resistances;
    std::vector<long double> theta;
    std::vector<long double> energy;
    std::vector<long double> semantic_charge;
    std::vector<long double> zeta_spectrum;
    std::vector<long double> padic_coordinates;
    std::vector<FieldChart> charts;
    std::uint64_t generation = 0;

    std::size_t size() const noexcept {
        return primes.size();
    }

    bool empty() const noexcept {
        return primes.empty();
    }

    void resize(std::size_t new_size) {
        primes.resize(new_size);
        phases.resize(new_size);
        activations.resize(new_size);
        resistances.resize(new_size);
        theta.resize(new_size);
        energy.resize(new_size);
        semantic_charge.resize(new_size);
        zeta_spectrum.resize(new_size);
        padic_coordinates.resize(new_size);
    }
};

inline long double field_unit_from_hash(std::uint64_t value) {
    return static_cast<long double>(value % 1'000'003ULL) / 1'000'003.0L;
}

inline std::vector<long double> field_impulse_signature(std::string_view text, std::size_t width) {
    std::vector<long double> signature(width, 0.0L);
    if (width == 0) {
        return signature;
    }
    const auto tokens = tokenize_query(text);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        std::uint64_t seed = stable_hash(tokens[i]) ^ (0x9e3779b97f4a7c15ULL * (i + 1U));
        for (std::size_t j = 0; j < width; ++j) {
            const long double wave = 2.0L * field_unit_from_hash(splitmix64(seed)) - 1.0L;
            signature[j] += wave / std::sqrt(static_cast<long double>(tokens.size()));
        }
    }
    const long double norm = std::sqrt(std::inner_product(signature.begin(), signature.end(),
                                                          signature.begin(), 0.0L));
    if (norm > 1.0e-18L) {
        for (auto& item : signature) {
            item /= norm;
        }
    }
    return signature;
}

inline long double field_cosine_similarity(const std::vector<long double>& left,
                                           const std::vector<long double>& right) {
    const std::size_t count = std::min(left.size(), right.size());
    if (count == 0) {
        return 0.0L;
    }
    long double dot = 0.0L;
    long double left_norm = 0.0L;
    long double right_norm = 0.0L;
    for (std::size_t i = 0; i < count; ++i) {
        dot += left[i] * right[i];
        left_norm += left[i] * left[i];
        right_norm += right[i] * right[i];
    }
    if (left_norm <= 1.0e-18L || right_norm <= 1.0e-18L) {
        return 0.0L;
    }
    return std::clamp(dot / std::sqrt(left_norm * right_norm), -1.0L, 1.0L);
}

inline std::vector<FieldChart> build_field_charts(const FieldState& field, std::size_t chart_count = 4) {
    std::vector<FieldChart> charts;
    if (field.empty()) {
        return charts;
    }
    chart_count = std::max<std::size_t>(1, std::min(chart_count, field.size()));
    charts.resize(chart_count);
    for (std::size_t i = 0; i < field.size(); ++i) {
        const std::size_t chart_index = static_cast<std::size_t>(field.primes[i] % chart_count);
        charts[chart_index].indices.push_back(i);
    }
    for (auto& chart : charts) {
        if (chart.indices.empty()) {
            continue;
        }
        chart.anchor_prime = field.primes[chart.indices.front()];
        long double phase_sum = 0.0L;
        long double activation_sum = 0.0L;
        long double locality_sum = 0.0L;
        for (auto index : chart.indices) {
            phase_sum += field.phases[index];
            activation_sum += field.activations[index];
            locality_sum += 1.0L / (1.0L + std::abs(field.padic_coordinates[index]));
        }
        const long double denom = static_cast<long double>(chart.indices.size());
        chart.mean_phase = phase_sum / denom;
        chart.mean_activation = activation_sum / denom;
        chart.locality = locality_sum / denom;
    }
    return charts;
}

inline FieldState make_field_state(const std::vector<Handle>& handles,
                                   std::string_view impulse,
                                   std::size_t limit = 0) {
    FieldState field;
    const std::size_t take = limit == 0 ? handles.size() : std::min(limit, handles.size());
    field.primes.reserve(take);
    field.phases.reserve(take);
    field.activations.reserve(take);
    field.resistances.reserve(take);
    field.theta.reserve(take);
    field.energy.reserve(take);
    field.padic_coordinates.reserve(take);
    field.zeta_spectrum.reserve(take);
    const auto signature = field_impulse_signature(impulse, take);
    field.semantic_charge = signature;

    for (std::size_t i = 0; i < take; ++i) {
        const auto& handle = handles[i];
        field.primes.push_back(handle.prime);
        field.phases.push_back(handle.phase);
        field.activations.push_back(handle.activation);
        field.resistances.push_back(handle.resistance);
        field.theta.push_back(handle.theta);
        field.energy.push_back(handle.energy);
        field.padic_coordinates.push_back(std::log1p(static_cast<long double>(handle.prime % 997U)) *
                                          (1.0L + signature[i]));
        field.zeta_spectrum.push_back(std::cos(handle.theta + handle.phase) *
                                      std::sqrt(std::max(0.0L, handle.energy) + 1.0L));
    }
    field.charts = build_field_charts(field);
    return field;
}

inline FieldState make_seed_field_state(std::string_view impulse, std::size_t count) {
    count = std::max<std::size_t>(1, count);
    const auto primes = generate_first_primes(count);
    FieldState field;
    field.primes = primes;
    field.phases.reserve(count);
    field.activations.reserve(count);
    field.resistances.reserve(count);
    field.theta.reserve(count);
    field.energy.reserve(count);
    field.padic_coordinates.reserve(count);
    field.zeta_spectrum.reserve(count);
    field.semantic_charge = field_impulse_signature(impulse, count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto prime = primes[i];
        const long double charge = field.semantic_charge[i];
        const long double phase = wrap_phase(riemann_siegel_theta(prime) + charge);
        field.phases.push_back(phase);
        field.activations.push_back(std::clamp(0.45L + 0.35L * std::abs(charge), 0.0L, 1.0L));
        field.resistances.push_back(std::clamp(0.55L - 0.10L * charge, 0.05L, 1.0L));
        field.theta.push_back(riemann_siegel_theta(prime));
        field.energy.push_back(spectral_energy(prime, 32));
        field.padic_coordinates.push_back(std::log1p(static_cast<long double>(prime % 997U)) * (1.0L + charge));
        field.zeta_spectrum.push_back(std::cos(phase + field.theta.back()) *
                                      std::sqrt(std::max(0.0L, field.energy.back()) + 1.0L));
    }
    field.charts = build_field_charts(field);
    return field;
}

inline void normalize_field_state(FieldState& field) {
    const std::size_t count = field.size();
    field.phases.resize(count);
    field.activations.resize(count, 0.0L);
    field.resistances.resize(count, 0.5L);
    field.theta.resize(count, 0.0L);
    field.energy.resize(count, 0.0L);
    field.semantic_charge.resize(count, 0.0L);
    field.zeta_spectrum.resize(count, 0.0L);
    field.padic_coordinates.resize(count, 0.0L);
    for (auto& phase : field.phases) {
        phase = wrap_phase(phase);
    }
    for (auto& activation : field.activations) {
        activation = std::clamp(activation, 0.0L, 1.0L);
    }
    for (auto& resistance : field.resistances) {
        resistance = std::clamp(resistance, 0.0L, 1.0L);
    }
    field.charts = build_field_charts(field);
}

inline void append_field_handle(FieldState& field,
                                std::uint32_t prime,
                                long double phase,
                                long double activation,
                                long double semantic_charge) {
    field.primes.push_back(prime);
    field.phases.push_back(wrap_phase(phase));
    field.activations.push_back(std::clamp(activation, 0.0L, 1.0L));
    field.resistances.push_back(0.5L);
    field.theta.push_back(riemann_siegel_theta(prime));
    field.energy.push_back(spectral_energy(prime, 32));
    field.semantic_charge.push_back(semantic_charge);
    field.zeta_spectrum.push_back(std::cos(field.phases.back() + field.theta.back()) *
                                  std::sqrt(std::max(0.0L, field.energy.back()) + 1.0L));
    field.padic_coordinates.push_back(std::log1p(static_cast<long double>(prime % 997U)) *
                                      (1.0L + semantic_charge));
    ++field.generation;
    normalize_field_state(field);
}

inline std::uint64_t field_geometry_signature(const FieldState& field) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < field.size(); ++i) {
        const auto phase_bucket = static_cast<std::uint64_t>((wrap_phase(field.phases[i]) + DZETA_PI) * 1000.0L);
        hash ^= field.primes[i] + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        hash ^= phase_bucket + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    }
    hash ^= field.generation + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    return hash;
}

} // namespace dzeta
