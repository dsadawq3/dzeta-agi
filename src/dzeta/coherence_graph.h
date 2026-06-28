#pragma once

#include "zeta_rhythm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace dzeta {

inline std::uint64_t turan_extremal_edges(std::size_t vertex_count, std::size_t clique_r) {
    if (clique_r <= 2 || vertex_count < clique_r) {
        return clique_r <= 2 ? 0ULL : static_cast<std::uint64_t>(vertex_count) *
                                       static_cast<std::uint64_t>(vertex_count - 1U) / 2ULL;
    }
    const std::size_t parts = clique_r - 1U;
    const std::size_t base = vertex_count / parts;
    const std::size_t extra = vertex_count % parts;
    std::vector<std::size_t> sizes(parts, base);
    for (std::size_t i = 0; i < extra; ++i) {
        ++sizes[i];
    }
    std::uint64_t internal = 0;
    for (auto size : sizes) {
        internal += static_cast<std::uint64_t>(size) * static_cast<std::uint64_t>(size - 1U) / 2ULL;
    }
    const std::uint64_t total = static_cast<std::uint64_t>(vertex_count) *
                                static_cast<std::uint64_t>(vertex_count - 1U) / 2ULL;
    return total - internal;
}

class CoherenceGraph {
public:
    explicit CoherenceGraph(std::size_t vertex_count = 0)
        : vertex_count_(vertex_count),
          words_per_row_((vertex_count + 63U) / 64U),
          adjacency_(vertex_count * words_per_row_, 0) {}

    std::size_t vertex_count() const noexcept {
        return vertex_count_;
    }

    void add_edge(std::size_t left, std::size_t right) {
        if (left == right || left >= vertex_count_ || right >= vertex_count_) {
            return;
        }
        set_bit(left, right);
        set_bit(right, left);
        ++edge_count;
        edge_density = possible_edges() == 0 ? 0.0L : static_cast<long double>(edge_count) /
                                                   static_cast<long double>(possible_edges());
    }

    bool connected(std::size_t left, std::size_t right) const {
        if (left >= vertex_count_ || right >= vertex_count_) {
            return false;
        }
        return (adjacency_[left * words_per_row_ + right / 64U] & (1ULL << (right % 64U))) != 0ULL;
    }

    std::size_t degree(std::size_t vertex) const {
        if (vertex >= vertex_count_) {
            return 0;
        }
        std::size_t degree_sum = 0;
        const std::size_t begin = vertex * words_per_row_;
        for (std::size_t word = 0; word < words_per_row_; ++word) {
            degree_sum += static_cast<std::size_t>(__builtin_popcountll(adjacency_[begin + word]));
        }
        return degree_sum;
    }

    long double turan_threshold(std::size_t clique_r) const {
        if (clique_r <= 2) {
            return 0.0L;
        }
        return 1.0L - 1.0L / static_cast<long double>(clique_r - 1U);
    }

    bool turan_guarantees_clique(std::size_t clique_r) const {
        if (vertex_count_ < clique_r || clique_r < 2) {
            return false;
        }
        return edge_density > turan_threshold(clique_r);
    }

    bool turan_guarantees_clique_exact(std::size_t clique_r) const {
        if (vertex_count_ < clique_r || clique_r < 2) {
            return false;
        }
        return static_cast<std::uint64_t>(edge_count) > turan_extremal_edges(vertex_count_, clique_r);
    }

    std::vector<std::size_t> find_clique(std::size_t clique_r) const {
        std::vector<std::size_t> clique;
        if (clique_r == 0 || vertex_count_ < clique_r) {
            return clique;
        }
        std::vector<std::size_t> candidates(vertex_count_);
        std::iota(candidates.begin(), candidates.end(), 0);
        find_clique_recursive(clique_r, clique, candidates);
        return clique.size() == clique_r ? clique : std::vector<std::size_t>{};
    }

    std::size_t max_clique_size(std::size_t search_limit = 128) const {
        if (vertex_count_ == 0) {
            return 0;
        }
        const std::size_t capped = std::min(vertex_count_, search_limit);
        if (capped != vertex_count_) {
            return greedy_clique_size(capped);
        }
        for (std::size_t r = vertex_count_; r >= 2; --r) {
            if (!find_clique(r).empty()) {
                return r;
            }
            if (r == 2) {
                break;
            }
        }
        return 1;
    }

    long double clique_pressure(std::size_t clique_r) const {
        if (clique_r < 2 || vertex_count_ < clique_r) {
            return 0.0L;
        }
        const long double exact_threshold = static_cast<long double>(turan_extremal_edges(vertex_count_, clique_r) + 1ULL);
        if (exact_threshold <= 0.0L) {
            return edge_count > 0 ? 1.0L : 0.0L;
        }
        const long double density_pressure = static_cast<long double>(edge_count) / exact_threshold;
        const long double witness_pressure = static_cast<long double>(max_clique_size()) /
                                             static_cast<long double>(clique_r);
        return std::clamp(std::max(density_pressure, witness_pressure), 0.0L, 2.0L);
    }

    std::uint64_t possible_edges() const {
        return vertex_count_ < 2 ? 0ULL : static_cast<std::uint64_t>(vertex_count_) *
                                      static_cast<std::uint64_t>(vertex_count_ - 1U) / 2ULL;
    }

    std::size_t edge_count = 0;
    long double edge_density = 0.0L;

private:
    void set_bit(std::size_t row, std::size_t column) {
        adjacency_[row * words_per_row_ + column / 64U] |= (1ULL << (column % 64U));
    }

    bool find_clique_recursive(std::size_t target,
                               std::vector<std::size_t>& clique,
                               const std::vector<std::size_t>& candidates) const {
        if (clique.size() == target) {
            return true;
        }
        if (clique.size() + candidates.size() < target) {
            return false;
        }

        std::vector<std::size_t> ordered = candidates;
        std::sort(ordered.begin(), ordered.end(), [&](std::size_t left, std::size_t right) {
            return degree(left) > degree(right);
        });

        for (std::size_t index = 0; index < ordered.size(); ++index) {
            const auto vertex = ordered[index];
            clique.push_back(vertex);
            std::vector<std::size_t> next;
            for (std::size_t j = index + 1; j < ordered.size(); ++j) {
                if (connected(vertex, ordered[j])) {
                    next.push_back(ordered[j]);
                }
            }
            if (find_clique_recursive(target, clique, next)) {
                return true;
            }
            clique.pop_back();
        }
        return false;
    }

    std::size_t greedy_clique_size(std::size_t limit) const {
        std::vector<std::size_t> vertices(limit);
        std::iota(vertices.begin(), vertices.end(), 0);
        std::sort(vertices.begin(), vertices.end(), [&](std::size_t left, std::size_t right) {
            return degree(left) > degree(right);
        });
        std::vector<std::size_t> clique;
        for (auto vertex : vertices) {
            bool ok = true;
            for (auto existing : clique) {
                if (!connected(vertex, existing)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                clique.push_back(vertex);
            }
        }
        return std::max<std::size_t>(1, clique.size());
    }

    std::size_t vertex_count_ = 0;
    std::size_t words_per_row_ = 0;
    std::vector<std::uint64_t> adjacency_;
};

inline CoherenceGraph build_coherence_graph(const std::vector<long double>& phases, long double epsilon) {
    CoherenceGraph graph(phases.size());
    for (std::size_t i = 0; i < phases.size(); ++i) {
        for (std::size_t j = i + 1; j < phases.size(); ++j) {
            if (phase_distance(phases[i], phases[j]) < epsilon) {
                graph.add_edge(i, j);
            }
        }
    }
    return graph;
}

} // namespace dzeta
