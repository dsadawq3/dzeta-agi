#pragma once

#include "coherence_graph.h"
#include "field_state.h"
#include "information.h"
#include "langlands.h"
#include "quantum_chaos.h"
#include "sle.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace dzeta {

struct FieldEnergyReport {
    long double zeta_coherence_energy = 0.0L;
    long double padic_compactness_energy = 0.0L;
    long double clique_energy = 0.0L;
    long double entropy_structure_energy = 0.0L;
    long double langlands_consistency_energy = 0.0L;
    long double sle_liveliness_energy = 0.0L;
    long double quantum_alignment_energy = 0.0L;
    long double memory_resonance_energy = 0.0L;
    long double syntax_energy = 0.0L;
    long double total_free_energy = 0.0L;
    long double stability = 0.0L;
    long double clique_pressure = 0.0L;
    long double edge_density = 0.0L;
    std::size_t max_clique = 0;
};

struct FieldAttractor {
    FieldEnergyReport initial_energy;
    FieldEnergyReport final_energy;
    std::vector<long double> trajectory;
    bool stable = false;
};

inline std::vector<std::size_t> field_phase_bins(const std::vector<long double>& phases,
                                                 std::size_t bins = 16) {
    std::vector<std::size_t> counts(std::max<std::size_t>(1, bins), 0);
    for (auto phase : phases) {
        const long double normalized = (wrap_phase(phase) + DZETA_PI) / (2.0L * DZETA_PI);
        const auto index = std::min<std::size_t>(counts.size() - 1U,
                                                 static_cast<std::size_t>(normalized * counts.size()));
        ++counts[index];
    }
    return counts;
}

inline long double field_padic_compactness(const FieldState& field) {
    if (field.size() < 2) {
        return 1.0L;
    }
    long double sum = 0.0L;
    std::size_t compared = 0;
    const std::size_t limit = std::min<std::size_t>(field.size(), 96);
    for (std::size_t i = 0; i < limit; ++i) {
        for (std::size_t j = i + 1; j < limit; ++j) {
            const long double gap = std::abs(field.padic_coordinates[i] - field.padic_coordinates[j]);
            sum += 1.0L / (1.0L + gap);
            ++compared;
        }
    }
    return compared == 0 ? 1.0L : sum / static_cast<long double>(compared);
}

inline long double field_zeta_alignment(const FieldState& field) {
    if (field.empty()) {
        return 0.0L;
    }
    long double sum = 0.0L;
    for (std::size_t i = 0; i < field.size(); ++i) {
        sum += std::abs(std::cos(field.phases[i] + field.theta[i])) *
               (0.5L + 0.5L * field.activations[i]);
    }
    return sum / static_cast<long double>(field.size());
}

inline std::vector<Handle> field_as_handles(const FieldState& field, std::size_t limit = 96) {
    std::vector<Handle> handles;
    const std::size_t count = std::min(limit, field.size());
    handles.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        Handle handle;
        handle.prime = field.primes[i];
        handle.phase = field.phases[i];
        handle.activation = field.activations[i];
        handle.resistance = field.resistances[i];
        handle.theta = field.theta[i];
        handle.energy = field.energy[i];
        handle.state = field.semantic_charge[i];
        handles.push_back(handle);
    }
    return handles;
}

inline FieldEnergyReport evaluate_variational_energy(const FieldState& field,
                                                     long double memory_resonance = 0.0L,
                                                     long double syntax_energy = 0.0L) {
    FieldEnergyReport report;
    if (field.empty()) {
        report.total_free_energy = 1.0L + syntax_energy;
        report.stability = 0.0L;
        return report;
    }

    const auto graph = build_coherence_graph(field.phases, 0.18L);
    report.edge_density = graph.edge_density;
    report.max_clique = graph.max_clique_size(96);
    report.clique_pressure = graph.clique_pressure(3);

    const long double zeta_alignment = field_zeta_alignment(field);
    const long double padic_compactness = field_padic_compactness(field);
    const long double entropy = shannon_entropy_bits(field_phase_bins(field.phases, 16));
    const long double normalized_entropy = entropy / 4.0L;
    const auto handles = field_as_handles(field, 64);
    const auto langlands = langlands_signature(handles);
    const auto sle = estimate_sle(field.phases, 16, field_geometry_signature(field));
    const auto chaos = quantum_chaos_report(graph, field.zeta_spectrum, 32);

    report.zeta_coherence_energy = 1.0L - std::clamp(zeta_alignment, 0.0L, 1.0L);
    report.padic_compactness_energy = 1.0L - std::clamp(padic_compactness, 0.0L, 1.0L);
    report.clique_energy = 1.0L / (1.0L + report.clique_pressure);
    report.entropy_structure_energy = std::clamp(normalized_entropy, 0.0L, 2.0L) *
                                      (1.0L - std::clamp(report.clique_pressure, 0.0L, 1.0L));
    report.langlands_consistency_energy = 1.0L / (1.0L + std::abs(langlands.global_norm) +
                                                 std::abs(langlands.euler_product_abs));
    report.sle_liveliness_energy = std::abs(sle.kappa - 4.0L) / 12.0L;
    report.quantum_alignment_energy = 1.0L - std::clamp(chaos.quantum_chaos_score, 0.0L, 1.0L);
    report.memory_resonance_energy = 1.0L - std::clamp(memory_resonance, 0.0L, 1.0L);
    report.syntax_energy = std::max(0.0L, syntax_energy);

    report.total_free_energy =
        0.20L * report.zeta_coherence_energy +
        0.16L * report.padic_compactness_energy +
        0.18L * report.clique_energy +
        0.14L * report.entropy_structure_energy +
        0.08L * report.langlands_consistency_energy +
        0.06L * report.sle_liveliness_energy +
        0.08L * report.quantum_alignment_energy +
        0.06L * report.memory_resonance_energy +
        0.04L * report.syntax_energy;
    report.stability = 1.0L / (1.0L + report.total_free_energy);
    return report;
}

inline void variational_step(FieldState& field, long double learning_rate = 0.08L) {
    if (field.empty()) {
        return;
    }
    learning_rate = std::clamp(learning_rate, 0.001L, 0.25L);
    long double phase_sin = 0.0L;
    long double phase_cos = 0.0L;
    long double charge_mean = 0.0L;
    for (std::size_t i = 0; i < field.size(); ++i) {
        const long double weight = 0.2L + field.activations[i];
        phase_sin += weight * std::sin(field.phases[i] + 0.25L * field.semantic_charge[i]);
        phase_cos += weight * std::cos(field.phases[i] + 0.25L * field.semantic_charge[i]);
        charge_mean += field.semantic_charge[i];
    }
    const long double target_phase = std::atan2(phase_sin, phase_cos);
    charge_mean /= static_cast<long double>(field.size());

    for (std::size_t i = 0; i < field.size(); ++i) {
        const long double theta_pull = wrap_phase(target_phase - field.phases[i]);
        const long double zeta_pull = std::sin(field.theta[i] + field.semantic_charge[i]);
        field.phases[i] = wrap_phase(field.phases[i] + learning_rate * theta_pull +
                                     0.025L * zeta_pull);
        const long double coherence_gain = 1.0L - std::min(1.0L, std::abs(theta_pull) / DZETA_PI);
        field.activations[i] = std::clamp(field.activations[i] + learning_rate * (coherence_gain - 0.45L),
                                          0.0L, 1.0L);
        field.resistances[i] = std::clamp(field.resistances[i] - 0.5L * learning_rate *
                                          (field.activations[i] - 0.35L), 0.02L, 1.0L);
        field.semantic_charge[i] = 0.98L * field.semantic_charge[i] + 0.02L * charge_mean;
        field.zeta_spectrum[i] = std::cos(field.phases[i] + field.theta[i]) *
                                 std::sqrt(std::max(0.0L, field.energy[i]) + 1.0L);
    }
    ++field.generation;
    normalize_field_state(field);
}

inline FieldAttractor run_variational_attractor(FieldState& field,
                                                std::size_t steps,
                                                long double memory_resonance = 0.0L,
                                                long double syntax_energy = 0.0L) {
    FieldAttractor attractor;
    normalize_field_state(field);
    attractor.initial_energy = evaluate_variational_energy(field, memory_resonance, syntax_energy);
    attractor.trajectory.push_back(attractor.initial_energy.total_free_energy);
    auto best_field = field;
    auto best_energy = attractor.initial_energy;

    for (std::size_t step = 0; step < steps; ++step) {
        auto candidate = field;
        variational_step(candidate, 0.10L / (1.0L + 0.05L * static_cast<long double>(step)));
        const auto energy = evaluate_variational_energy(candidate, memory_resonance, syntax_energy);
        if (energy.total_free_energy <= best_energy.total_free_energy) {
            field = candidate;
            best_field = candidate;
            best_energy = energy;
        } else {
            variational_step(field, 0.025L);
            const auto fallback = evaluate_variational_energy(field, memory_resonance, syntax_energy);
            if (fallback.total_free_energy < best_energy.total_free_energy) {
                best_field = field;
                best_energy = fallback;
            }
        }
        attractor.trajectory.push_back(best_energy.total_free_energy);
    }
    field = best_field;
    attractor.final_energy = best_energy;
    attractor.stable = attractor.final_energy.stability >= attractor.initial_energy.stability ||
                       attractor.final_energy.total_free_energy <= attractor.initial_energy.total_free_energy;
    return attractor;
}

} // namespace dzeta
