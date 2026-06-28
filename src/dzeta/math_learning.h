#pragma once

#include "semantic_field_memory.h"

#include <algorithm>
#include <string>
#include <string_view>

namespace dzeta {

struct FieldLearningReport {
    std::size_t attractor_index = 0;
    long double basin_strength_before = 0.0L;
    long double basin_strength_after = 0.0L;
    long double entropy_penalty_before = 0.0L;
    long double entropy_penalty_after = 0.0L;
    FieldEnergyReport energy;
};

inline FieldLearningReport learn_field_example(SemanticFieldMemory& memory,
                                               FieldState& field,
                                               std::string_view prompt,
                                               std::string_view output,
                                               bool success) {
    const auto hits = memory.resonate(prompt, field, 1);
    FieldLearningReport report;
    if (!hits.empty()) {
        report.attractor_index = hits.front().index;
        report.basin_strength_before = memory.at(hits.front().index).strength;
        report.entropy_penalty_before = memory.at(hits.front().index).entropy_penalty;
    }

    const auto syntax_penalty = success ? 0.0L : 2.0L;
    auto attractor = run_variational_attractor(field, success ? 10 : 3, hits.empty() ? 0.0L : hits.front().score,
                                               syntax_penalty);
    report.energy = attractor.final_energy;
    const auto index = memory.imprint(prompt, field, report.energy, std::string(output), success);
    report.attractor_index = index;

    auto& stored = memory.at(index);
    if (success) {
        stored.strength = std::clamp(stored.strength + 0.35L * attractor.final_energy.stability, 0.0L, 16.0L);
        stored.entropy_penalty = std::max(0.0L, stored.entropy_penalty - 0.08L);
    } else {
        stored.strength = std::max(0.0L, stored.strength - 0.12L);
        stored.entropy_penalty = std::clamp(stored.entropy_penalty + 0.55L, 0.0L, 8.0L);
    }
    report.basin_strength_after = stored.strength;
    report.entropy_penalty_after = stored.entropy_penalty;
    return report;
}

inline long double field_memory_resonance_score(const SemanticFieldMemory& memory,
                                                std::string_view prompt,
                                                const FieldState& field) {
    const auto hits = memory.resonate(prompt, field, 1);
    return hits.empty() ? 0.0L : hits.front().score;
}

} // namespace dzeta
