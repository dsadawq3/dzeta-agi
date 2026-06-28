#pragma once

#include "differentiable_field.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace dzeta {

struct AdamWFieldOptimizer {
    DifferentiableFieldParameters m;
    DifferentiableFieldParameters v;
    long double learning_rate = 0.01L;
    long double beta1 = 0.9L;
    long double beta2 = 0.999L;
    long double weight_decay = 0.001L;
    long double epsilon = 1.0e-8L;
    std::size_t step_count = 0;

    explicit AdamWFieldOptimizer(const DifferentiableFieldParameters& params, long double lr = 0.01L)
        : m(zero_gradient_like(params)), v(zero_gradient_like(params)), learning_rate(lr) {}
};

inline long double gradient_l2_norm(const DifferentiableFieldGradient& grad) {
    long double sum = 0.0L;
    for (const auto* vector : {&grad.phase_coupling, &grad.padic_chart_weights, &grad.zeta_coupling,
                               &grad.memory_basin_weights, &grad.syntax_weights}) {
        for (auto value : *vector) {
            sum += value * value;
        }
    }
    return std::sqrt(sum);
}

inline void clip_gradient(DifferentiableFieldGradient& grad, long double max_norm) {
    const long double norm = gradient_l2_norm(grad);
    if (norm <= max_norm || norm <= 1.0e-18L) {
        return;
    }
    const long double scale = max_norm / norm;
    for (auto* vector : {&grad.phase_coupling, &grad.padic_chart_weights, &grad.zeta_coupling,
                         &grad.memory_basin_weights, &grad.syntax_weights}) {
        for (auto& value : *vector) {
            value *= scale;
        }
    }
}

inline void adamw_update_vector(std::vector<long double>& param,
                                std::vector<long double>& m,
                                std::vector<long double>& v,
                                const std::vector<long double>& grad,
                                AdamWFieldOptimizer& optimizer) {
    const long double b1_correction = 1.0L - std::pow(optimizer.beta1, static_cast<long double>(optimizer.step_count));
    const long double b2_correction = 1.0L - std::pow(optimizer.beta2, static_cast<long double>(optimizer.step_count));
    for (std::size_t i = 0; i < param.size() && i < grad.size(); ++i) {
        m[i] = optimizer.beta1 * m[i] + (1.0L - optimizer.beta1) * grad[i];
        v[i] = optimizer.beta2 * v[i] + (1.0L - optimizer.beta2) * grad[i] * grad[i];
        const long double m_hat = m[i] / std::max(optimizer.epsilon, b1_correction);
        const long double v_hat = v[i] / std::max(optimizer.epsilon, b2_correction);
        param[i] -= optimizer.learning_rate * (m_hat / (std::sqrt(v_hat) + optimizer.epsilon) +
                                               optimizer.weight_decay * param[i]);
    }
}

inline void optimizer_step(DifferentiableFieldParameters& params,
                           AdamWFieldOptimizer& optimizer,
                           DifferentiableFieldGradient grad,
                           long double max_grad_norm = 1.0L) {
    ++optimizer.step_count;
    clip_gradient(grad, max_grad_norm);
    adamw_update_vector(params.phase_coupling, optimizer.m.phase_coupling, optimizer.v.phase_coupling,
                        grad.phase_coupling, optimizer);
    adamw_update_vector(params.padic_chart_weights, optimizer.m.padic_chart_weights, optimizer.v.padic_chart_weights,
                        grad.padic_chart_weights, optimizer);
    adamw_update_vector(params.zeta_coupling, optimizer.m.zeta_coupling, optimizer.v.zeta_coupling,
                        grad.zeta_coupling, optimizer);
    adamw_update_vector(params.memory_basin_weights, optimizer.m.memory_basin_weights, optimizer.v.memory_basin_weights,
                        grad.memory_basin_weights, optimizer);
    adamw_update_vector(params.syntax_weights, optimizer.m.syntax_weights, optimizer.v.syntax_weights,
                        grad.syntax_weights, optimizer);
    clamp_differentiable_params(params);
}

} // namespace dzeta
