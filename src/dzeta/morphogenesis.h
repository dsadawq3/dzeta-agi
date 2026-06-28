#pragma once

#include "semantic_field_memory.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <unordered_set>
#include <vector>

namespace dzeta {

struct MorphogenesisReport {
    std::size_t born_handles = 0;
    std::size_t merged_handles = 0;
    std::size_t split_clusters = 0;
    std::size_t new_charts = 0;
    std::size_t memory_organs = 0;
    std::size_t unstable_removed = 0;
    std::size_t merged_pairs = 0;
    std::size_t pruned_dead = 0;
    long double structural_entropy_before = 0.0L;
    long double structural_entropy_after = 0.0L;
    bool changed = false;
};

inline std::uint32_t next_field_prime(const FieldState& field) {
    const auto primes = generate_first_primes(field.size() + 32);
    for (auto prime : primes) {
        if (std::find(field.primes.begin(), field.primes.end(), prime) == field.primes.end()) {
            return prime;
        }
    }
    return primes.empty() ? 2U : primes.back();
}

inline long double morphogenesis_structural_entropy(const FieldState& field) {
    if (field.size() < 2) {
        return 0.0L;
    }
    std::vector<long double> phase_bins(16, 0.0L);
    for (const auto p : field.phases) {
        const auto idx = std::min<std::size_t>(15, static_cast<std::size_t>((p + DZETA_PI) / DZETA_TWO_PI * 16.0L));
        phase_bins[idx] += 1.0L;
    }
    long double entropy = 0.0L;
    const long double n = static_cast<long double>(field.size());
    for (const auto count : phase_bins) {
        if (count <= 0.0L) continue;
        const long double p = count / n;
        entropy -= p * std::log(p);
    }
    return entropy / std::log(2.0L);
}

inline void merge_similar_handles(FieldState& field, long double phase_threshold, MorphogenesisReport& report) {
    if (field.size() < 2) {
        return;
    }
    std::vector<bool> merged(field.size(), false);
    for (std::size_t i = 0; i < field.size(); ++i) {
        if (merged[i]) continue;
        for (std::size_t j = i + 1; j < field.size(); ++j) {
            if (merged[j]) continue;
            const long double phase_gap = phase_distance(field.phases[i], field.phases[j]);
            const long double padic_gap = std::abs(field.padic_coordinates[i] - field.padic_coordinates[j]);
            if (phase_gap < phase_threshold && padic_gap < 0.15L) {
                const long double total_act = field.activations[i] + field.activations[j];
                field.activations[i] = std::min(1.0L, total_act);
                field.semantic_charge[i] = 0.5L * (field.semantic_charge[i] + field.semantic_charge[j]);
                field.energy[i] = std::max(field.energy[i], field.energy[j]);
                merged[j] = true;
                ++report.merged_handles;
                ++report.merged_pairs;
                report.changed = true;
            }
        }
    }
    if (report.merged_pairs > 0) {
        std::size_t write = 0;
        for (std::size_t i = 0; i < field.size(); ++i) {
            if (!merged[i]) {
                if (write != i) {
                    field.primes[write] = field.primes[i];
                    field.phases[write] = field.phases[i];
                    field.activations[write] = field.activations[i];
                    field.resistances[write] = field.resistances[i];
                    field.theta[write] = field.theta[i];
                    field.energy[write] = field.energy[i];
                    field.semantic_charge[write] = field.semantic_charge[i];
                    field.padic_coordinates[write] = field.padic_coordinates[i];
                    field.zeta_spectrum[write] = field.zeta_spectrum[i];
                }
                ++write;
            }
        }
        field.resize(write);
    }
}

inline void prune_unstable_handles(FieldState& field, long double min_activation, MorphogenesisReport& report) {
    std::size_t pruned = 0;
    for (std::size_t i = 0; i < field.size(); ) {
        if (field.activations[i] < min_activation && field.resistances[i] > 0.92L) {
            ++pruned;
            if (i + 1 < field.size()) {
                field.primes[i] = field.primes.back();
                field.phases[i] = field.phases.back();
                field.activations[i] = field.activations.back();
                field.resistances[i] = field.resistances.back();
                field.theta[i] = field.theta.back();
                field.energy[i] = field.energy.back();
                field.semantic_charge[i] = field.semantic_charge.back();
                field.padic_coordinates[i] = field.padic_coordinates.back();
                field.zeta_spectrum[i] = field.zeta_spectrum.back();
            }
            field.resize(field.size() - 1);
        } else {
            ++i;
        }
    }
    report.pruned_dead = pruned;
    if (pruned > 0) {
        report.changed = true;
    }
}

inline void split_clusters_by_phase(FieldState& field, long double coherence_threshold, MorphogenesisReport& report) {
    if (field.size() < 4) {
        return;
    }
    std::vector<long double> phase_ref = field.phases;
    std::sort(phase_ref.begin(), phase_ref.end());
    std::vector<std::size_t> gaps;
    for (std::size_t i = 1; i < phase_ref.size(); ++i) {
        if (std::abs(phase_ref[i] - phase_ref[i - 1]) > coherence_threshold) {
            gaps.push_back(i);
        }
    }
    if (gaps.size() < 2) {
        return;
    }
    report.split_clusters += gaps.size() / 2U;
    report.new_charts += gaps.size() / 2U;
    report.changed = true;
}

inline MorphogenesisReport apply_morphogenesis(FieldState& field,
                                               SemanticFieldMemory& memory,
                                               const FieldEnergyReport& energy,
                                               bool allow_growth) {
    MorphogenesisReport report;
    if (field.empty()) {
        field = make_seed_field_state("morphogenesis seed", 8);
        report.born_handles = field.size();
        report.new_charts = field.charts.size();
        report.changed = true;
        return report;
    }

    report.structural_entropy_before = morphogenesis_structural_entropy(field);

    merge_similar_handles(field, 0.12L, report);

    prune_unstable_handles(field, 0.02L, report);

    if (allow_growth && (energy.total_free_energy > 0.75L || energy.stability < 0.35L)) {
        const std::size_t births = std::max<std::size_t>(1, std::min<std::size_t>(8, field.size() / 8U + 1U));
        for (std::size_t i = 0; i < births; ++i) {
            const auto prime = next_field_prime(field);
            const long double charge = (i % 2 == 0 ? 0.35L : -0.35L) / static_cast<long double>(i + 1U);
            const long double phase_offset = DZETA_TWO_PI * static_cast<long double>(i) / static_cast<long double>(births);
            append_field_handle(field,
                                prime,
                                energy.total_free_energy + charge + 0.1L * std::sin(phase_offset),
                                0.35L + 0.05L * static_cast<long double>(i % 3U),
                                charge);
            ++report.born_handles;
        }
        ++report.new_charts;
        report.changed = true;
    }

    if (energy.total_free_energy > 0.60L && energy.total_free_energy < 0.80L) {
        split_clusters_by_phase(field, 1.8L, report);
    }

    if (energy.stability > 0.62L && energy.total_free_energy < 0.55L) {
        memory.imprint("morphogenesis stable organ", field, energy, "memory-organ", true);
        ++report.memory_organs;
        report.changed = true;
    }

    if (report.changed) {
        report.structural_entropy_after = morphogenesis_structural_entropy(field);
    }

    field.charts = build_field_charts(field, std::max<std::size_t>(4, field.charts.size() + report.new_charts));
    ++field.generation;
    return report;
}

} // namespace dzeta
