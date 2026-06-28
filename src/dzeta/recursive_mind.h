#pragma once

#include "mentalese_core.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace dzeta {

struct MentalAnchor {
    MentalState prompt_anchor;
};

struct RecursiveMindConfig {
    std::size_t max_iterations = 8;
    std::size_t min_iterations = 2;
    long double min_energy_delta = 0.0001L;
    long double anchor_injection = 0.14L;
    long double target_uncertainty = 0.10L;
};

struct RecursiveLoopReport {
    MentalState final_state;
    std::size_t iterations = 0;
    std::vector<long double> energy_history;
    bool halted = false;
    long double anchor_drift = 0.0L;
};

inline long double recursive_mind_energy(const MentalState& state,
                                         const MentalAnchor& anchor) {
    const long double similarity = mental_similarity(state, anchor.prompt_anchor);
    const long double structure_bonus = state.has_symbol("recursive_refined") ? 0.08L : 0.0L;
    return std::clamp(state.uncertainty + 0.35L * (1.0L - similarity) - structure_bonus, 0.0L, 2.0L);
}

class RecursiveMentalLoop {
public:
    explicit RecursiveMentalLoop(RecursiveMindConfig config = {})
        : config_(config) {}

    RecursiveLoopReport refine(const MentalState& initial,
                               const MentalAnchor& anchor) const {
        RecursiveLoopReport report;
        report.final_state = initial;
        if (report.final_state.vectors.empty()) {
            report.final_state.vectors = anchor.prompt_anchor.vectors;
        }
        long double previous_energy = recursive_mind_energy(report.final_state, anchor);
        report.energy_history.push_back(previous_energy);
        const std::size_t max_iterations = std::max<std::size_t>(1, config_.max_iterations);
        const long double injection = std::clamp(config_.anchor_injection, 0.0L, 0.95L);
        for (std::size_t step = 0; step < max_iterations; ++step) {
            const std::size_t n = std::min(report.final_state.vectors.size(), anchor.prompt_anchor.vectors.size());
            for (std::size_t i = 0; i < n; ++i) {
                const long double phase = ((static_cast<long double>((i + 1U) * (step + 3U) % 17U) / 17.0L) - 0.5L) * 0.012L;
                report.final_state.vectors[i] =
                    (1.0L - injection) * report.final_state.vectors[i] +
                    injection * anchor.prompt_anchor.vectors[i] + phase;
            }
            normalize_vector(report.final_state.vectors);
            report.final_state.uncertainty =
                std::max(config_.target_uncertainty,
                         report.final_state.uncertainty * (0.72L - 0.02L * std::min<std::size_t>(step, 5U)));
            mental_add_symbol(report.final_state, "recursive_refined");
            mental_add_symbol(report.final_state, "latent_loop");
            ++report.iterations;
            const long double energy = recursive_mind_energy(report.final_state, anchor);
            report.energy_history.push_back(std::min(energy, previous_energy));
            const long double delta = previous_energy - report.energy_history.back();
            previous_energy = report.energy_history.back();
            const bool can_halt = report.iterations >= std::min(config_.min_iterations, max_iterations);
            if (can_halt &&
                (report.final_state.uncertainty <= config_.target_uncertainty ||
                 (step > 0 && delta >= 0.0L && delta < config_.min_energy_delta))) {
                report.halted = true;
                break;
            }
        }
        if (!report.halted) {
            report.halted = report.iterations == max_iterations;
        }
        report.anchor_drift = vector_drift(report.final_state.vectors, anchor.prompt_anchor.vectors);
        return report;
    }

private:
    static long double vector_drift(const std::vector<long double>& left,
                                    const std::vector<long double>& right) {
        const std::size_t n = std::min(left.size(), right.size());
        if (n == 0) {
            return 1.0L;
        }
        long double dot = 0.0L;
        long double left_norm = 0.0L;
        long double right_norm = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            dot += left[i] * right[i];
            left_norm += left[i] * left[i];
            right_norm += right[i] * right[i];
        }
        if (left_norm <= 1.0e-12L || right_norm <= 1.0e-12L) {
            return 1.0L;
        }
        const long double cosine = dot / (std::sqrt(left_norm) * std::sqrt(right_norm));
        return std::clamp(1.0L - ((cosine + 1.0L) * 0.5L), 0.0L, 1.0L);
    }

    static void normalize_vector(std::vector<long double>& vector) {
        const long double norm = std::sqrt(std::inner_product(vector.begin(), vector.end(), vector.begin(), 0.0L));
        if (norm <= 1.0e-12L) {
            return;
        }
        for (auto& value : vector) {
            value /= norm;
        }
    }

    RecursiveMindConfig config_;
};

} // namespace dzeta
