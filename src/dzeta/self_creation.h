#pragma once

#include "math_learning.h"
#include "morphogenesis.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct SelfCreationReport {
    bool changed_geometry = false;
    std::size_t expanded_charts = 0;
    std::size_t compressed_basins = 0;
    std::size_t concept_organs = 0;
    std::size_t new_concepts = 0;
    std::size_t memory_snapshots = 0;
    std::uint64_t before_signature = 0;
    std::uint64_t after_signature = 0;
};

inline std::uint64_t field_signature(const FieldState& field) {
    if (field.empty()) {
        return 0;
    }
    std::uint64_t sig = field.generation;
    for (std::size_t i = 0; i < std::min<std::size_t>(field.size(), 32U); ++i) {
        sig ^= stable_hash(std::to_string(field.primes[i]) + ":" +
                           std::to_string(static_cast<double>(field.phases[i])) + ":" +
                           std::to_string(static_cast<double>(field.activations[i])));
        sig = splitmix64(sig);
    }
    return sig;
}

inline std::vector<std::string> infer_concepts_from_field(const FieldState& field,
                                                           const SemanticFieldMemory& memory) {
    std::vector<std::string> concept_labels;
    if (field.size() < 3) {
        return concept_labels;
    }
    std::vector<long double> activation_sorted = field.activations;
    std::sort(activation_sorted.begin(), activation_sorted.end(), std::greater<long double>());
    const long double threshold = activation_sorted.size() >= 3U
                                      ? activation_sorted[field.size() / 3U]
                                      : 0.3L;
    for (std::size_t i = 0; i < field.size(); ++i) {
        if (field.activations[i] >= threshold) {
            concept_labels.push_back("concept_" + std::to_string(field.primes[i]));
        }
    }
    (void)memory;
    return concept_labels;
}

inline SelfCreationReport self_create_from_failure(FieldState& field,
                                                   SemanticFieldMemory& memory,
                                                   std::string_view prompt,
                                                   std::string_view output,
                                                   const FieldEnergyReport& failure_energy) {
    SelfCreationReport report;
    report.before_signature = field_signature(field);
    auto adjusted_energy = failure_energy;
    adjusted_energy.total_free_energy = std::max(adjusted_energy.total_free_energy, 1.1L);
    adjusted_energy.stability = std::min(adjusted_energy.stability, 0.25L);
    const auto morph = apply_morphogenesis(field, memory, adjusted_energy, true);

    const auto detected_concepts = infer_concepts_from_field(field, memory);
    for (const auto& cpt : detected_concepts) {
        memory.imprint(prompt, field, failure_energy, cpt, false);
        ++report.new_concepts;
        ++report.memory_snapshots;
    }

    learn_field_example(memory, field, prompt, output, false);
    report.expanded_charts = morph.new_charts;
    report.concept_organs = morph.memory_organs;
    report.after_signature = field_signature(field);
    report.changed_geometry = report.before_signature != report.after_signature || morph.changed;
    return report;
}

inline SelfCreationReport self_create_from_success(FieldState& field,
                                                   SemanticFieldMemory& memory,
                                                   std::string_view prompt,
                                                   std::string_view output) {
    SelfCreationReport report;
    report.before_signature = field_signature(field);
    const auto learning = learn_field_example(memory, field, prompt, output, true);
    auto energy = learning.energy;
    energy.stability = std::max(energy.stability, 0.70L);
    energy.total_free_energy = std::min(energy.total_free_energy, 0.40L);
    const auto morph = apply_morphogenesis(field, memory, energy, false);

    const auto detected_concepts = infer_concepts_from_field(field, memory);
    for (const auto& cpt : detected_concepts) {
        memory.imprint(prompt, field, energy, cpt, true);
        ++report.new_concepts;
        ++report.memory_snapshots;
    }

    report.compressed_basins = learning.basin_strength_after > learning.basin_strength_before ? 1U : 0U;
    report.concept_organs = morph.memory_organs;
    report.after_signature = field_signature(field);
    report.changed_geometry = report.before_signature != report.after_signature || morph.changed;
    return report;
}

inline SelfCreationReport self_create_continuous(FieldState& field,
                                                  SemanticFieldMemory& memory,
                                                  std::string_view context,
                                                  const FieldEnergyReport& current_energy,
                                                  std::size_t max_new_organs = 3) {
    SelfCreationReport report;
    report.before_signature = field_signature(field);

    if (current_energy.stability > 0.55L && current_energy.total_free_energy < 0.65L) {
        const auto detected_concepts = infer_concepts_from_field(field, memory);
        std::size_t created = 0;
        for (const auto& cpt : detected_concepts) {
            if (created >= max_new_organs) break;
            memory.imprint(context, field, current_energy, cpt, true);
            ++created;
            ++report.new_concepts;
            ++report.memory_snapshots;
        }
        report.concept_organs += created;
    }

    if (current_energy.stability > 0.75L) {
        const auto morph = apply_morphogenesis(field, memory, current_energy, false);
        report.compressed_basins += morph.merged_handles;
    }

    report.after_signature = field_signature(field);
    report.changed_geometry = report.before_signature != report.after_signature;
    return report;
}

} // namespace dzeta
