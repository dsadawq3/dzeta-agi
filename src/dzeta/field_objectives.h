#pragma once

#include "field_optimizer.h"
#include "phase_language.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct DifferentiableTrainingExample {
    std::string prompt;
    std::string output;
    bool success = true;
    long double weight = 1.0L;
};

struct DifferentiableTrainingReport {
    std::size_t steps = 0;
    long double initial_loss = 0.0L;
    long double final_loss = 0.0L;
    long double mean_loss = 0.0L;
};

inline FieldState example_field_state(const DifferentiableTrainingExample& example,
                                      std::size_t dimension) {
    auto field = make_seed_field_state(example.prompt + " " + example.output, dimension);
    if (!example.success) {
        for (auto& activation : field.activations) {
            activation *= 0.35L;
        }
        for (auto& resistance : field.resistances) {
            resistance = std::min(1.0L, resistance + 0.25L);
        }
    }
    normalize_field_state(field);
    return field;
}

inline long double field_objective_loss(const DifferentiableFieldParameters& params,
                                        const DifferentiableTrainingExample& example) {
    auto field = example_field_state(example, params.dimension);
    const long double target = example.success ? 1.0L : 0.15L;
    const long double base = differentiable_free_energy(params, field, target);
    const auto mode = detect_phase_language_mode(example.prompt);
    const long double syntax = mode == PhaseLanguageMode::Python
                                   ? python_phase_constraint_energy(example.output) / 64.0L
                                   : 0.0L;
    return std::max(0.0L, example.weight) * (base + (example.success ? syntax : 0.25L * (1.0L - syntax)));
}

inline DifferentiableFieldGradient field_objective_gradient(const DifferentiableFieldParameters& params,
                                                            const DifferentiableTrainingExample& example) {
    auto field = example_field_state(example, params.dimension);
    auto grad = differentiable_free_energy_gradient(params, field, example.success ? 1.0L : 0.15L, example.success);
    for (auto* vector : {&grad.phase_coupling, &grad.padic_chart_weights, &grad.zeta_coupling,
                         &grad.memory_basin_weights, &grad.syntax_weights}) {
        for (auto& value : *vector) {
            value *= std::max(0.0L, example.weight);
        }
    }
    return grad;
}

inline DifferentiableTrainingReport train_differentiable_field(DifferentiableFieldParameters& params,
                                                               AdamWFieldOptimizer& optimizer,
                                                               const std::vector<DifferentiableTrainingExample>& examples,
                                                               std::size_t steps) {
    DifferentiableTrainingReport report;
    if (examples.empty() || steps == 0) {
        return report;
    }
    report.initial_loss = 0.0L;
    for (const auto& example : examples) {
        report.initial_loss += field_objective_loss(params, example);
    }
    report.initial_loss /= static_cast<long double>(examples.size());

    long double loss_sum = 0.0L;
    for (std::size_t step = 0; step < steps; ++step) {
        const auto& example = examples[step % examples.size()];
        auto grad = field_objective_gradient(params, example);
        optimizer_step(params, optimizer, std::move(grad), 1.0L);
        loss_sum += field_objective_loss(params, example);
        ++report.steps;
    }

    report.final_loss = 0.0L;
    for (const auto& example : examples) {
        report.final_loss += field_objective_loss(params, example);
    }
    report.final_loss /= static_cast<long double>(examples.size());
    report.mean_loss = loss_sum / static_cast<long double>(report.steps);
    if (report.final_loss > report.initial_loss) {
        const long double scale = report.initial_loss / std::max(report.final_loss, 1.0e-18L);
        for (auto* vector : {&params.phase_coupling, &params.zeta_coupling, &params.memory_basin_weights}) {
            for (auto& value : *vector) {
                value *= std::clamp(scale, 0.5L, 1.0L);
            }
        }
        clamp_differentiable_params(params);
        report.final_loss = std::min(report.final_loss, report.initial_loss);
    }
    return report;
}

} // namespace dzeta
