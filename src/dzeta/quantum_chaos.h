#pragma once

#include "coherence_graph.h"
#include "zeta_zeros.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace dzeta {

struct QuantumChaosReport {
    std::vector<long double> eigenvalues;
    long double mean_spacing = 0.0L;
    long double spacing_variance = 0.0L;
    long double wigner_distance = 1.0L;
    long double poisson_distance = 1.0L;
    long double zeta_spacing_correlation = 0.0L;
    long double quantum_chaos_score = 0.0L;
};

inline std::vector<long double> jacobi_eigenvalues(std::vector<std::vector<long double>> matrix) {
    const std::size_t n = matrix.size();
    if (n == 0) {
        return {};
    }
    const std::size_t max_iterations = 64U * n * n;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        std::size_t p = 0;
        std::size_t q = 1;
        long double max_off = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const auto value = std::abs(matrix[i][j]);
                if (value > max_off) {
                    max_off = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < 1.0e-12L || n < 2) {
            break;
        }
        const long double phi = 0.5L * std::atan2(2.0L * matrix[p][q], matrix[q][q] - matrix[p][p]);
        const long double c = std::cos(phi);
        const long double s = std::sin(phi);
        for (std::size_t k = 0; k < n; ++k) {
            const long double mkp = matrix[k][p];
            const long double mkq = matrix[k][q];
            matrix[k][p] = c * mkp - s * mkq;
            matrix[k][q] = s * mkp + c * mkq;
        }
        for (std::size_t k = 0; k < n; ++k) {
            const long double mpk = matrix[p][k];
            const long double mqk = matrix[q][k];
            matrix[p][k] = c * mpk - s * mqk;
            matrix[q][k] = s * mpk + c * mqk;
        }
    }
    std::vector<long double> eigenvalues(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = matrix[i][i];
    }
    std::sort(eigenvalues.begin(), eigenvalues.end());
    return eigenvalues;
}

inline QuantumChaosReport quantum_chaos_report(const CoherenceGraph& graph,
                                               const std::vector<long double>& potential,
                                               std::size_t zero_count) {
    QuantumChaosReport report;
    const std::size_t n = std::min<std::size_t>(graph.vertex_count(), potential.size());
    if (n == 0) {
        return report;
    }
    std::vector<std::vector<long double>> hamiltonian(n, std::vector<long double>(n, 0.0L));
    for (std::size_t i = 0; i < n; ++i) {
        const long double degree_i = static_cast<long double>(std::max<std::size_t>(1, graph.degree(i)));
        hamiltonian[i][i] = 1.0L + potential[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            if (!graph.connected(i, j)) {
                continue;
            }
            const long double degree_j = static_cast<long double>(std::max<std::size_t>(1, graph.degree(j)));
            const long double value = -1.0L / std::sqrt(degree_i * degree_j);
            hamiltonian[i][j] = value;
            hamiltonian[j][i] = value;
        }
    }

    report.eigenvalues = jacobi_eigenvalues(std::move(hamiltonian));
    if (report.eigenvalues.size() < 3) {
        report.quantum_chaos_score = 0.5L;
        return report;
    }

    std::vector<long double> gaps;
    for (std::size_t i = 1; i < report.eigenvalues.size(); ++i) {
        const auto gap = report.eigenvalues[i] - report.eigenvalues[i - 1];
        if (gap > 1.0e-14L) {
            gaps.push_back(gap);
        }
    }
    if (gaps.empty()) {
        return report;
    }
    for (auto gap : gaps) {
        report.mean_spacing += gap;
    }
    report.mean_spacing /= static_cast<long double>(gaps.size());
    for (auto& gap : gaps) {
        gap /= report.mean_spacing;
    }
    long double variance = 0.0L;
    for (auto gap : gaps) {
        const auto delta = gap - 1.0L;
        variance += delta * delta;
    }
    report.spacing_variance = variance / static_cast<long double>(gaps.size());
    constexpr long double wigner_variance = 0.2732395447351627L;
    constexpr long double poisson_variance = 1.0L;
    report.wigner_distance = std::abs(report.spacing_variance - wigner_variance);
    report.poisson_distance = std::abs(report.spacing_variance - poisson_variance);

    const std::size_t compare = std::min({gaps.size(), zero_count, zeta_zero_count() - 1U});
    if (compare > 1) {
        long double dot = 0.0L;
        long double left_norm = 0.0L;
        long double right_norm = 0.0L;
        for (std::size_t i = 0; i < compare; ++i) {
            const long double zgap = normalized_zero_gap(i);
            dot += gaps[i] * zgap;
            left_norm += gaps[i] * gaps[i];
            right_norm += zgap * zgap;
        }
        if (left_norm > 0.0L && right_norm > 0.0L) {
            report.zeta_spacing_correlation = std::clamp(dot / std::sqrt(left_norm * right_norm), -1.0L, 1.0L);
        }
    }
    const long double wigner_score = 1.0L / (1.0L + report.wigner_distance);
    const long double zeta_score = 0.5L * (1.0L + report.zeta_spacing_correlation);
    report.quantum_chaos_score = std::clamp(0.65L * wigner_score + 0.35L * zeta_score, 0.0L, 1.0L);
    return report;
}

} // namespace dzeta
