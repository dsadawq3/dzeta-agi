#pragma once

#include "field_state.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct DifferentiableFieldParameters {
    std::size_t dimension = 0;
    std::size_t charts = 0;
    std::vector<long double> phase_coupling;
    std::vector<long double> padic_chart_weights;
    std::vector<long double> zeta_coupling;
    std::vector<long double> memory_basin_weights;
    std::vector<long double> syntax_weights;
    std::vector<long double> routing_weights;
    std::vector<long double> memory_gates;
    std::vector<long double> planner_policy;
    std::vector<long double> causal_edges;
    std::vector<long double> decoder_logits;
    std::vector<long double> module_confidence;
};

using DifferentiableFieldGradient = DifferentiableFieldParameters;

inline DifferentiableFieldParameters make_differentiable_field_parameters(std::size_t dimension,
                                                                          std::size_t charts,
                                                                          std::uint64_t seed) {
    dimension = std::max<std::size_t>(1, dimension);
    charts = std::max<std::size_t>(1, charts);
    DifferentiableFieldParameters params;
    params.dimension = dimension;
    params.charts = charts;
    params.phase_coupling.resize(dimension);
    params.padic_chart_weights.resize(charts);
    params.zeta_coupling.resize(dimension);
    params.memory_basin_weights.resize(dimension);
    params.syntax_weights.resize(8);
    params.routing_weights.resize(dimension);
    params.memory_gates.resize(dimension);
    params.planner_policy.resize(dimension);
    params.causal_edges.resize(dimension);
    params.decoder_logits.resize(dimension);
    params.module_confidence.resize(dimension);
    for (auto* vector : {&params.phase_coupling, &params.zeta_coupling, &params.memory_basin_weights,
                         &params.routing_weights, &params.memory_gates, &params.planner_policy,
                         &params.causal_edges, &params.decoder_logits, &params.module_confidence}) {
        for (auto& value : *vector) {
            value = 0.20L + 0.20L * field_unit_from_hash(splitmix64(seed));
        }
    }
    for (auto& value : params.padic_chart_weights) {
        value = 0.25L + 0.15L * field_unit_from_hash(splitmix64(seed));
    }
    for (auto& value : params.syntax_weights) {
        value = 0.10L + 0.10L * field_unit_from_hash(splitmix64(seed));
    }
    return params;
}

inline DifferentiableFieldGradient zero_gradient_like(const DifferentiableFieldParameters& params) {
    DifferentiableFieldGradient grad = params;
    std::fill(grad.phase_coupling.begin(), grad.phase_coupling.end(), 0.0L);
    std::fill(grad.padic_chart_weights.begin(), grad.padic_chart_weights.end(), 0.0L);
    std::fill(grad.zeta_coupling.begin(), grad.zeta_coupling.end(), 0.0L);
    std::fill(grad.memory_basin_weights.begin(), grad.memory_basin_weights.end(), 0.0L);
    std::fill(grad.syntax_weights.begin(), grad.syntax_weights.end(), 0.0L);
    std::fill(grad.routing_weights.begin(), grad.routing_weights.end(), 0.0L);
    std::fill(grad.memory_gates.begin(), grad.memory_gates.end(), 0.0L);
    std::fill(grad.planner_policy.begin(), grad.planner_policy.end(), 0.0L);
    std::fill(grad.causal_edges.begin(), grad.causal_edges.end(), 0.0L);
    std::fill(grad.decoder_logits.begin(), grad.decoder_logits.end(), 0.0L);
    std::fill(grad.module_confidence.begin(), grad.module_confidence.end(), 0.0L);
    return grad;
}

inline long double differentiable_free_energy(const DifferentiableFieldParameters& params,
                                              const FieldState& field,
                                              long double target_stability = 1.0L) {
    if (field.empty()) {
        return 1.0L;
    }
    long double energy = 0.0L;
    long double weight_sum = 0.0L;
    for (std::size_t i = 0; i < field.size(); ++i) {
        const std::size_t dim = i % std::max<std::size_t>(1, params.dimension);
        const std::size_t chart = static_cast<std::size_t>(field.primes[i] % std::max<std::size_t>(1, params.charts));
        const long double phase_w = params.phase_coupling.empty() ? 0.2L : params.phase_coupling[dim];
        const long double zeta_w = params.zeta_coupling.empty() ? 0.2L : params.zeta_coupling[dim];
        const long double memory_w = params.memory_basin_weights.empty() ? 0.2L : params.memory_basin_weights[dim];
        const long double chart_w = params.padic_chart_weights.empty() ? 0.2L : params.padic_chart_weights[chart];
        const long double zeta_alignment = std::abs(std::cos(field.phases[i] + field.theta[i]));
        const long double activation_gap = std::abs(target_stability - field.activations[i]);
        const long double charge = std::abs(field.semantic_charge[i]);
        energy += phase_w * activation_gap +
                  zeta_w * (1.0L - zeta_alignment) +
                  chart_w / (1.0L + std::abs(field.padic_coordinates[i])) +
                  memory_w * (1.0L - std::min(1.0L, charge));
        weight_sum += phase_w + zeta_w + chart_w + memory_w;
    }
    const long double syntax = params.syntax_weights.empty()
                                   ? 0.0L
                                   : std::accumulate(params.syntax_weights.begin(),
                                                     params.syntax_weights.end(),
                                                     0.0L) / static_cast<long double>(params.syntax_weights.size());
    long double param_sum = 0.0L;
    std::size_t param_count = 0;
    for (const auto* vector : {&params.phase_coupling, &params.padic_chart_weights, &params.zeta_coupling,
                               &params.memory_basin_weights, &params.syntax_weights, &params.routing_weights,
                               &params.memory_gates, &params.planner_policy, &params.causal_edges,
                               &params.decoder_logits, &params.module_confidence}) {
        param_sum += std::accumulate(vector->begin(), vector->end(), 0.0L);
        param_count += vector->size();
    }
    const long double param_pressure = param_count == 0
                                           ? 0.0L
                                           : param_sum / static_cast<long double>(param_count);
    return energy / std::max(1.0L, weight_sum) + 0.01L * syntax + 0.002L * param_pressure;
}

inline DifferentiableFieldGradient differentiable_free_energy_gradient(const DifferentiableFieldParameters& params,
                                                                       const FieldState& field,
                                                                       long double target_stability,
                                                                       bool success) {
    auto grad = zero_gradient_like(params);
    if (field.empty()) {
        return grad;
    }
    const long double direction = success ? 1.0L : -1.0L;
    for (std::size_t i = 0; i < field.size(); ++i) {
        const std::size_t dim = i % std::max<std::size_t>(1, params.dimension);
        const std::size_t chart = static_cast<std::size_t>(field.primes[i] % std::max<std::size_t>(1, params.charts));
        const long double zeta_alignment = std::abs(std::cos(field.phases[i] + field.theta[i]));
        const long double activation_gap = std::abs(target_stability - field.activations[i]);
        const long double charge = std::abs(field.semantic_charge[i]);
        grad.phase_coupling[dim] += direction * activation_gap / static_cast<long double>(field.size());
        grad.zeta_coupling[dim] += direction * (1.0L - zeta_alignment) / static_cast<long double>(field.size());
        grad.memory_basin_weights[dim] += direction * (1.0L - std::min(1.0L, charge)) /
                                           static_cast<long double>(field.size());
        grad.padic_chart_weights[chart] += direction /
                                            ((1.0L + std::abs(field.padic_coordinates[i])) *
                                             static_cast<long double>(field.size()));
    }
    for (auto& value : grad.syntax_weights) {
        value = success ? 0.012L : -0.04L;
    }
    for (auto* vector : {&grad.routing_weights, &grad.memory_gates, &grad.planner_policy,
                         &grad.causal_edges, &grad.decoder_logits, &grad.module_confidence}) {
        for (auto& value : *vector) {
            value = success ? 0.01L : -0.02L;
        }
    }
    return grad;
}

inline void clamp_differentiable_params(DifferentiableFieldParameters& params) {
    for (auto* vector : {&params.phase_coupling, &params.padic_chart_weights, &params.zeta_coupling,
                         &params.memory_basin_weights, &params.syntax_weights, &params.routing_weights,
                         &params.memory_gates, &params.planner_policy, &params.causal_edges,
                         &params.decoder_logits, &params.module_confidence}) {
        for (auto& value : *vector) {
            value = std::clamp(value, 0.001L, 4.0L);
        }
    }
}

inline void save_differentiable_field_checkpoint(const DifferentiableFieldParameters& params,
                                                 std::string_view path) {
    std::ofstream output(std::string(path), std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write differentiable field checkpoint: " + std::string(path));
    }
    const auto write_vec = [&](std::string_view name, const std::vector<long double>& values) {
        output << name << '\t' << values.size();
        for (auto value : values) {
            output << '\t' << static_cast<double>(value);
        }
        output << '\n';
    };
    output << "DZETA_DIFFERENTIABLE_FIELD_V1\t" << params.dimension << '\t' << params.charts << '\n';
    write_vec("phase", params.phase_coupling);
    write_vec("padic", params.padic_chart_weights);
    write_vec("zeta", params.zeta_coupling);
    write_vec("memory", params.memory_basin_weights);
    write_vec("syntax", params.syntax_weights);
    write_vec("routing", params.routing_weights);
    write_vec("memory_gates", params.memory_gates);
    write_vec("planner_policy", params.planner_policy);
    write_vec("causal_edges", params.causal_edges);
    write_vec("decoder_logits", params.decoder_logits);
    write_vec("module_confidence", params.module_confidence);
}

inline DifferentiableFieldParameters load_differentiable_field_checkpoint(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read differentiable field checkpoint: " + std::string(path));
    }
    std::string line;
    std::getline(input, line);
    std::stringstream header(line);
    std::string magic;
    std::string item;
    std::getline(header, magic, '\t');
    if (magic != "DZETA_DIFFERENTIABLE_FIELD_V1") {
        throw std::runtime_error("invalid differentiable field checkpoint: " + std::string(path));
    }
    std::getline(header, item, '\t');
    DifferentiableFieldParameters params;
    params.dimension = static_cast<std::size_t>(std::stoull(item));
    std::getline(header, item, '\t');
    params.charts = static_cast<std::size_t>(std::stoull(item));
    const auto read_vec = [](const std::string& row) {
        std::stringstream stream(row);
        std::string name;
        std::string field;
        std::getline(stream, name, '\t');
        std::getline(stream, field, '\t');
        const auto count = static_cast<std::size_t>(std::stoull(field));
        std::vector<long double> values;
        values.reserve(count);
        while (std::getline(stream, field, '\t')) {
            values.push_back(std::stold(field));
        }
        values.resize(count, 0.0L);
        return std::pair<std::string, std::vector<long double>>(name, values);
    };
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        auto [name, values] = read_vec(line);
        if (name == "phase") {
            params.phase_coupling = std::move(values);
        } else if (name == "padic") {
            params.padic_chart_weights = std::move(values);
        } else if (name == "zeta") {
            params.zeta_coupling = std::move(values);
        } else if (name == "memory") {
            params.memory_basin_weights = std::move(values);
        } else if (name == "syntax") {
            params.syntax_weights = std::move(values);
        } else if (name == "routing") {
            params.routing_weights = std::move(values);
        } else if (name == "memory_gates") {
            params.memory_gates = std::move(values);
        } else if (name == "planner_policy") {
            params.planner_policy = std::move(values);
        } else if (name == "causal_edges") {
            params.causal_edges = std::move(values);
        } else if (name == "decoder_logits") {
            params.decoder_logits = std::move(values);
        } else if (name == "module_confidence") {
            params.module_confidence = std::move(values);
        }
    }
    if (params.routing_weights.empty()) {
        params.routing_weights.assign(params.dimension, 0.2L);
    }
    if (params.memory_gates.empty()) {
        params.memory_gates.assign(params.dimension, 0.2L);
    }
    if (params.planner_policy.empty()) {
        params.planner_policy.assign(params.dimension, 0.2L);
    }
    if (params.causal_edges.empty()) {
        params.causal_edges.assign(params.dimension, 0.2L);
    }
    if (params.decoder_logits.empty()) {
        params.decoder_logits.assign(params.dimension, 0.2L);
    }
    if (params.module_confidence.empty()) {
        params.module_confidence.assign(params.dimension, 0.2L);
    }
    return params;
}

} // namespace dzeta
