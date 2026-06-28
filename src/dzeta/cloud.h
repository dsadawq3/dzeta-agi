#pragma once

#include "coherence_graph.h"
#include "cpu_accel.h"
#include "dissipative_adaptation.h"
#include "entropy.h"
#include "entropy_singularity.h"
#include "handle.h"
#include "information.h"
#include "kerner.h"
#include "langlands.h"
#include "padic.h"
#include "primes.h"
#include "quantum_chaos.h"
#include "sat.h"
#include "sle.h"
#include "zeta_universality.h"
#include "zeta_rhythm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct DzetaConfig {
    std::size_t handle_count = 1'000'000;
    std::size_t active_window = 4096;
    std::size_t zeros_per_tick = 256;
    long double temperature = 0.05L;
    long double coherence_epsilon = 0.12L;
    long double threshold_F_to_S = 0.18L;
    long double threshold_S_to_J = 0.42L;
    long double max_resistance = 1.0L;
    long double base_resistance = 0.52L;
    long double sigmoid_gain = 6.0L;
    long double twin_transfer_rate = 0.03L;
    long double invariant_tolerance = 0.33L;
    long double invariant_min_score = 0.5L;
    bool require_rdrand = true;
    bool math_core_enabled = false;
    bool strict_math = false;
    bool print_math = false;
    std::size_t target_clique_r = 3;
    std::size_t min_survival_clique = 3;
    std::size_t sle_steps = 16;
    std::size_t zeta_samples = 64;
    std::size_t spectral_window = 32;
    std::size_t clique_search_limit = 128;
    CpuDispatchMode cpu_dispatch = CpuDispatchMode::Auto;
};

struct CloudSnapshot {
    KernerPhase phase = KernerPhase::FreeFlow;
    long double coherent_density = 0.0L;
    long double mean_activation = 0.0L;
    long double mean_energy = 0.0L;
    std::size_t active_begin = 0;
    std::size_t active_count = 0;
    std::uint64_t answer_seed = 0;
    long double shannon_entropy = 0.0L;
    long double mutual_information = 0.0L;
    long double kolmogorov_upper_bound = 0.0L;
    long double edge_density = 0.0L;
    long double turan_threshold = 0.0L;
    std::size_t guaranteed_clique_r = 0;
    std::size_t max_clique_found = 0;
    long double entropy_singularity = 0.0L;
    long double dissipative_adaptation = 0.0L;
    long double sle_kappa = 0.0L;
    long double quantum_chaos_score = 0.0L;
    long double zeta_symmetry_residual = 0.0L;
    long double universality_score = 0.0L;
    long double langlands_score = 0.0L;
};

inline long double sigmoid(long double value) {
    if (value > 40.0L) {
        return 1.0L;
    }
    if (value < -40.0L) {
        return 0.0L;
    }
    return 1.0L / (1.0L + std::exp(-value));
}

class Cloud {
public:
    Cloud(std::string name, DzetaConfig config, std::uint64_t seed)
        : name_(std::move(name)),
          config_(config),
          entropy_(seed, config.require_rdrand ? EntropyMode::RequireRdrand : EntropyMode::AllowFallback),
          seed_(seed) {
        if (config_.handle_count == 0) {
            config_.handle_count = 1;
        }
        config_.active_window = std::max<std::size_t>(1, std::min(config_.active_window, config_.handle_count));
        config_.zeros_per_tick = std::max<std::size_t>(1, std::min(config_.zeros_per_tick, zeta_zero_count()));

        const auto primes = generate_first_primes(config_.handle_count);
        handles_.reserve(primes.size());
        for (auto prime : primes) {
            Handle handle;
            handle.prime = prime;
            handle.twin_prime = twin_prime_of(prime);
            handle.resistance = config_.base_resistance;
            handle.energy = spectral_energy(handle.prime, config_.zeros_per_tick);
            handle.theta = riemann_siegel_theta(handle.prime);
            handle.tuned = true;
            handles_.push_back(handle);
        }
    }

    const std::string& name() const noexcept {
        return name_;
    }

    const DzetaConfig& config() const noexcept {
        return config_;
    }

    std::vector<Handle>& handles() noexcept {
        return handles_;
    }

    const std::vector<Handle>& handles() const noexcept {
        return handles_;
    }

    KernerPhase phase() const noexcept {
        return phase_;
    }

    CloudSnapshot last_snapshot() const noexcept {
        return last_snapshot_;
    }

    std::string_view entropy_provider() const {
        return entropy_.provider();
    }

    CloudSnapshot tick(std::string_view impulse, long double tick_time) {
        if (handles_.empty()) {
            return {};
        }

        const auto problem = sat_from_query(impulse);
        const auto spectrum = complexity_spectrum(problem, 6, 24, stable_hash(impulse) ^ seed_);
        const long double dynamic_temperature = std::max(config_.temperature, spectrum.temperature_hint);

        const std::uint64_t hash = stable_hash(impulse) ^ seed_;
        const std::size_t range = handles_.size() - config_.active_window + 1;
        active_begin_ = range == 0 ? 0 : static_cast<std::size_t>(hash % range);
        const std::size_t active_end = active_begin_ + config_.active_window;

        std::vector<int> phase_bins(64, 0);
        std::vector<std::size_t> phase_bin_series;
        std::vector<std::size_t> activation_bin_series;
        std::vector<long double> active_phases;
        std::vector<long double> active_activations;
        std::vector<long double> active_potential;
        if (config_.math_core_enabled) {
            phase_bin_series.reserve(config_.active_window);
            activation_bin_series.reserve(config_.active_window);
            active_phases.reserve(config_.active_window);
            active_activations.reserve(config_.active_window);
            active_potential.reserve(config_.active_window);
        }
        long double activation_sum = 0.0L;
        long double energy_sum = 0.0L;
        std::uint64_t answer_seed = hash;

        for (std::size_t index = active_begin_; index < active_end; ++index) {
            auto& handle = handles_[index];
            if (handle.blocked) {
                handle.activation = 0.0L;
                continue;
            }

            handle.phase = zeta_phase(handle.prime, tick_time, config_.zeros_per_tick);
            const long double energy_norm = handle.energy / static_cast<long double>(config_.zeros_per_tick);
            const long double drive = std::cos(handle.theta +
                                               static_cast<long double>((hash % 1'000'003ULL) + 1ULL) *
                                                   0.000001L * static_cast<long double>(handle.prime));
            const long double noise = entropy_.signed_unit() * dynamic_temperature;
            handle.state = 0.86L * handle.state + 0.09L * drive + 0.05L * energy_norm + noise;
            handle.activation = sigmoid(config_.sigmoid_gain * (handle.state - handle.resistance));
            handle.resistance = std::clamp(handle.resistance + 0.015L * handle.activation - 0.006L,
                                           0.05L,
                                           config_.max_resistance);

            const long double normalized = (wrap_phase(handle.phase) + DZETA_PI) / DZETA_TWO_PI;
            const auto bin = std::clamp<int>(static_cast<int>(normalized * phase_bins.size()), 0,
                                             static_cast<int>(phase_bins.size() - 1));
            ++phase_bins[static_cast<std::size_t>(bin)];
            if (config_.math_core_enabled) {
                const auto phase_bin = static_cast<std::size_t>(bin);
                const auto activation_bin = std::clamp<std::size_t>(
                    static_cast<std::size_t>(handle.activation * 16.0L), 0, 15);
                phase_bin_series.push_back(phase_bin);
                activation_bin_series.push_back(activation_bin);
                active_phases.push_back(handle.phase);
                active_activations.push_back(handle.activation);
                active_potential.push_back(std::sqrt(std::max(0.0L, handle.energy)) /
                                           static_cast<long double>(config_.zeros_per_tick + 1U));
            }

            activation_sum += handle.activation;
            energy_sum += handle.energy;
            answer_seed ^= static_cast<std::uint64_t>(handle.activation * 1'000'000.0L + handle.prime);
            answer_seed = splitmix64(answer_seed);
        }

        transfer_twin_energy(active_begin_, active_end);

        const auto max_bin = *std::max_element(phase_bins.begin(), phase_bins.end());
        const long double coherent_density = static_cast<long double>(max_bin) /
                                             static_cast<long double>(config_.active_window);
        long double effective_density = coherent_density;

        CloudSnapshot math_snapshot;
        if (config_.math_core_enabled) {
            const std::vector<std::size_t> phase_counts(phase_bins.begin(), phase_bins.end());
            math_snapshot.shannon_entropy = shannon_entropy_bits(phase_counts);
            const auto normalized_phase_entropy = normalized_entropy_bits(phase_counts);
            math_snapshot.mutual_information = mutual_information_bits(phase_bin_series, activation_bin_series);
            const auto complexity = kolmogorov_upper_bound(impulse);
            math_snapshot.kolmogorov_upper_bound = complexity.normalized_complexity;

            const auto graph = build_coherence_graph(active_phases, config_.coherence_epsilon);
            math_snapshot.edge_density = graph.edge_density;
            math_snapshot.turan_threshold = graph.turan_threshold(config_.target_clique_r);
            math_snapshot.max_clique_found = graph.max_clique_size(config_.clique_search_limit);
            math_snapshot.guaranteed_clique_r = graph.turan_guarantees_clique_exact(config_.target_clique_r)
                                                    ? config_.target_clique_r
                                                    : math_snapshot.max_clique_found;
            const long double target_r = static_cast<long double>(std::max<std::size_t>(2, config_.target_clique_r));
            const long double clique_pressure = std::clamp(graph.clique_pressure(config_.target_clique_r), 0.0L, 1.0L);
            const long double mi_norm = std::clamp(math_snapshot.mutual_information /
                                                       std::max(1.0L, std::log2(target_r + 1.0L)),
                                                   0.0L,
                                                   1.0L);
            math_snapshot.entropy_singularity = entropy_singularity_score(normalized_phase_entropy,
                                                                           clique_pressure,
                                                                           mi_norm,
                                                                           1.0L);
            long double stability_gain = activation_sum / static_cast<long double>(config_.active_window);
            if (previous_activations_.size() == active_activations.size() && !active_activations.empty()) {
                long double dot = 0.0L;
                long double left_norm = 0.0L;
                long double right_norm = 0.0L;
                for (std::size_t i = 0; i < active_activations.size(); ++i) {
                    dot += previous_activations_[i] * active_activations[i];
                    left_norm += previous_activations_[i] * previous_activations_[i];
                    right_norm += active_activations[i] * active_activations[i];
                }
                if (left_norm > 0.0L && right_norm > 0.0L) {
                    stability_gain = std::max(stability_gain, dot / std::sqrt(left_norm * right_norm));
                }
            }
            previous_activations_ = active_activations;
            math_snapshot.dissipative_adaptation = dissipative_adaptation_score(previous_entropy_,
                                                                                math_snapshot.shannon_entropy,
                                                                                dynamic_temperature,
                                                                                stability_gain);
            previous_entropy_ = math_snapshot.shannon_entropy;

            const auto sle = estimate_sle(active_phases, config_.sle_steps, hash ^ seed_);
            math_snapshot.sle_kappa = sle.kappa;

            const std::size_t spectral_count = std::min(config_.spectral_window, active_phases.size());
            std::vector<long double> spectral_phases(active_phases.begin(), active_phases.begin() + spectral_count);
            std::vector<long double> spectral_potential(active_potential.begin(), active_potential.begin() + spectral_count);
            const auto spectral_graph = build_coherence_graph(spectral_phases, config_.coherence_epsilon);
            const auto chaos = quantum_chaos_report(spectral_graph, spectral_potential, config_.zeta_samples);
            math_snapshot.quantum_chaos_score = chaos.quantum_chaos_score;

            const auto zeta_report = zeta_functional_equation_report({2.0L, 0.5L + tick_time}, config_.zeta_samples);
            math_snapshot.zeta_symmetry_residual = zeta_report.functional_residual;
            math_snapshot.universality_score = zeta_report.universality_score;

            const auto langlands = langlands_signature(active_handles(config_.spectral_window));
            math_snapshot.langlands_score = std::clamp(langlands.global_norm, 0.0L, 1.0L);

            if (math_snapshot.entropy_singularity > 0.72L && !config_.strict_math) {
                config_.temperature = std::max(0.005L, config_.temperature * 0.98L);
                config_.coherence_epsilon = std::min(0.35L, config_.coherence_epsilon * 1.015L);
                config_.active_window = std::min(config_.handle_count,
                                                 config_.active_window + std::max<std::size_t>(1, config_.active_window / 64U));
            }
            if (math_snapshot.dissipative_adaptation > 0.05L) {
                for (std::size_t index = active_begin_; index < active_end && index < handles_.size(); ++index) {
                    auto& handle = handles_[index];
                    handle.resistance = std::clamp(handle.resistance -
                                                       0.004L * math_snapshot.dissipative_adaptation *
                                                           handle.activation,
                                                   0.05L,
                                                   config_.max_resistance);
                }
            }

            effective_density = std::max({effective_density,
                                          0.35L * math_snapshot.edge_density + 0.65L * clique_pressure,
                                          mi_norm});
        }

        phase_ = transition_phase(effective_density, config_.threshold_F_to_S, config_.threshold_S_to_J);

        last_snapshot_ = CloudSnapshot{
            phase_,
            coherent_density,
            activation_sum / static_cast<long double>(config_.active_window),
            energy_sum / static_cast<long double>(config_.active_window),
            active_begin_,
            config_.active_window,
            answer_seed,
        };
        if (config_.math_core_enabled) {
            last_snapshot_.shannon_entropy = math_snapshot.shannon_entropy;
            last_snapshot_.mutual_information = math_snapshot.mutual_information;
            last_snapshot_.kolmogorov_upper_bound = math_snapshot.kolmogorov_upper_bound;
            last_snapshot_.edge_density = math_snapshot.edge_density;
            last_snapshot_.turan_threshold = math_snapshot.turan_threshold;
            last_snapshot_.guaranteed_clique_r = math_snapshot.guaranteed_clique_r;
            last_snapshot_.max_clique_found = math_snapshot.max_clique_found;
            last_snapshot_.entropy_singularity = math_snapshot.entropy_singularity;
            last_snapshot_.dissipative_adaptation = math_snapshot.dissipative_adaptation;
            last_snapshot_.sle_kappa = math_snapshot.sle_kappa;
            last_snapshot_.quantum_chaos_score = math_snapshot.quantum_chaos_score;
            last_snapshot_.zeta_symmetry_residual = math_snapshot.zeta_symmetry_residual;
            last_snapshot_.universality_score = math_snapshot.universality_score;
            last_snapshot_.langlands_score = math_snapshot.langlands_score;
        }
        return last_snapshot_;
    }

    std::vector<long double> state_projection(std::size_t limit) const {
        std::vector<long double> projection;
        if (handles_.empty() || limit == 0) {
            return projection;
        }
        const std::size_t count = std::min({limit, config_.active_window, handles_.size()});
        projection.reserve(count * 3);
        const std::size_t begin = std::min(active_begin_, handles_.size() - count);
        for (std::size_t offset = 0; offset < count; ++offset) {
            const auto& handle = handles_[begin + offset];
            projection.push_back(handle.state);
            projection.push_back(handle.activation);
            projection.push_back(handle.phase);
        }
        return projection;
    }

    std::vector<Handle> active_handles(std::size_t limit) const {
        std::vector<Handle> result;
        if (handles_.empty() || limit == 0) {
            return result;
        }
        const std::size_t count = std::min({limit, config_.active_window, handles_.size()});
        result.reserve(count);
        const std::size_t begin = std::min(active_begin_, handles_.size() - count);
        for (std::size_t offset = 0; offset < count; ++offset) {
            result.push_back(handles_[begin + offset]);
        }
        return result;
    }

private:
    void transfer_twin_energy(std::size_t active_begin, std::size_t active_end) {
        if (config_.twin_transfer_rate <= 0.0L) {
            return;
        }
        for (std::size_t index = active_begin; index < active_end; ++index) {
            auto& handle = handles_[index];
            if (handle.twin_prime == 0 || handle.activation <= 0.0L) {
                continue;
            }
            const auto found = std::lower_bound(handles_.begin(), handles_.end(), handle.twin_prime,
                                                [](const Handle& item, std::uint32_t prime) {
                                                    return item.prime < prime;
                                                });
            if (found != handles_.end() && found->prime == handle.twin_prime) {
                const long double transfer = handle.state * config_.twin_transfer_rate;
                found->state += transfer;
                handle.state -= transfer;
            }
        }
    }

    std::string name_;
    DzetaConfig config_;
    HardwareEntropy entropy_;
    std::uint64_t seed_ = 0;
    std::vector<Handle> handles_;
    KernerPhase phase_ = KernerPhase::FreeFlow;
    CloudSnapshot last_snapshot_;
    std::size_t active_begin_ = 0;
    long double previous_entropy_ = 0.0L;
    std::vector<long double> previous_activations_;
};

} // namespace dzeta
