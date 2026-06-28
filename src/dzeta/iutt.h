#pragma once

#include "cloud.h"
#include "padic.h"
#include "sat.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct TheaterFrame {
    std::string name;
    std::vector<Handle> handles;
    std::vector<long double> theta_channel;
    std::vector<long double> log_shell;
    std::vector<long double> height_profile;
    std::vector<long double> frobenioid_profile;
    std::vector<std::string> bridge_keys;
    std::vector<int> xor_indeterminacy;
    long double log_volume = 0.0L;
    long double arithmetic_height = 0.0L;
    long double determinant_log = 0.0L;
    long double langlands_score = 0.0L;
    long double zeta_symmetry_residual = 0.0L;
    long double quantum_chaos_score = 0.0L;
    KernerPhase phase = KernerPhase::FreeFlow;
    std::uint64_t seed = 0;
};

struct BridgeReport {
    std::string from;
    std::string to;
    long double padic_distortion = 0.0L;
    long double theta_link_distance = 0.0L;
    long double theta_correlation = 0.0L;
    long double log_shell_xor_distance = 0.0L;
    long double log_volume_distance = 0.0L;
    long double height_gap = 0.0L;
    long double frobenioid_norm = 0.0L;
    long double determinant_gap = 0.0L;
    long double langlands_gap = 0.0L;
    long double zeta_symmetry_gap = 0.0L;
    long double quantum_chaos_gap = 0.0L;
    std::size_t compared_pairs = 0;
    std::size_t matched_pairs = 0;
    long double match_coverage = 0.0L;
    long double combined = 0.0L;
};

struct AnswerInvariant {
    bool accepted = false;
    long double score = 0.0L;
    std::string reason;
};

struct IuttResult {
    CloudSnapshot snapshot_a;
    CloudSnapshot snapshot_b;
    CloudSnapshot snapshot_c;
    BridgeReport bridge_ab;
    BridgeReport bridge_ac;
    BridgeReport bridge_bc;
    AnswerInvariant invariant;
    bool accepted = false;
    std::string answer_a;
    std::string answer_b;
    std::string answer_c;
    std::uint64_t seed = 0;
};

inline TheaterFrame make_theater_frame(const Cloud& cloud, std::size_t limit = 160) {
    TheaterFrame frame;
    frame.name = cloud.name();
    frame.handles = cloud.active_handles(limit);
    frame.phase = cloud.phase();
    frame.seed = cloud.last_snapshot().answer_seed;
    frame.langlands_score = cloud.last_snapshot().langlands_score;
    frame.zeta_symmetry_residual = cloud.last_snapshot().zeta_symmetry_residual;
    frame.quantum_chaos_score = cloud.last_snapshot().quantum_chaos_score;
    frame.theta_channel.reserve(frame.handles.size());
    frame.log_shell.reserve(frame.handles.size());
    frame.height_profile.reserve(frame.handles.size());
    frame.frobenioid_profile.reserve(frame.handles.size());
    frame.bridge_keys.reserve(frame.handles.size());
    frame.xor_indeterminacy.reserve(frame.handles.size());

    const auto name_hash = stable_hash(frame.name);
    for (const auto& handle : frame.handles) {
        const long double signed_state = handle.state >= 0.0L ? 1.0L : -1.0L;
        const long double energy_scale = std::sqrt(std::max(0.0L, handle.energy) + 1.0L);
        const long double local_height = std::log(static_cast<long double>(handle.prime)) *
                                         (0.25L + handle.activation);
        const long double theta_value = signed_state * std::log1p(std::abs(handle.state)) *
                                        energy_scale * std::cos(handle.phase + handle.theta);
        const long double shell_value = std::log1p(std::abs(handle.state));
        const long double frobenioid_value = std::log1p(static_cast<long double>(handle.prime)) *
                                             std::exp(-std::abs(wrap_phase(handle.phase))) *
                                             (1.0L + handle.activation);
        const int xor_bit = static_cast<int>((stable_hash(std::to_string(handle.prime)) ^ name_hash) & 1ULL);

        frame.theta_channel.push_back(theta_value);
        frame.log_shell.push_back(shell_value);
        frame.height_profile.push_back(local_height);
        frame.frobenioid_profile.push_back(frobenioid_value);
        frame.bridge_keys.push_back(padic_hierarchy_key(handle.prime, {2, 3, 5}, 1));
        frame.xor_indeterminacy.push_back(xor_bit);
        frame.arithmetic_height += local_height;
        frame.log_volume += std::log1p(std::abs(handle.state) * (energy_scale + handle.activation));
        frame.determinant_log += std::log1p(std::abs(theta_value) + shell_value + frobenioid_value);
    }
    return frame;
}

inline std::vector<std::pair<std::size_t, std::size_t>> keyed_bridge_pairs(const TheaterFrame& from,
                                                                            const TheaterFrame& to,
                                                                            bool& used_exact_keys) {
    struct Entry {
        std::string key;
        std::size_t index = 0;
    };

    std::vector<Entry> left;
    std::vector<Entry> right;
    left.reserve(from.bridge_keys.size());
    right.reserve(to.bridge_keys.size());
    for (std::size_t i = 0; i < from.bridge_keys.size(); ++i) {
        left.push_back({from.bridge_keys[i], i});
    }
    for (std::size_t i = 0; i < to.bridge_keys.size(); ++i) {
        right.push_back({to.bridge_keys[i], i});
    }
    const auto by_key = [](const Entry& a, const Entry& b) {
        if (a.key == b.key) {
            return a.index < b.index;
        }
        return a.key < b.key;
    };
    std::sort(left.begin(), left.end(), by_key);
    std::sort(right.begin(), right.end(), by_key);

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i].key == right[j].key) {
            pairs.emplace_back(left[i].index, right[j].index);
            ++i;
            ++j;
        } else if (left[i].key < right[j].key) {
            ++i;
        } else {
            ++j;
        }
    }

    used_exact_keys = !pairs.empty();
    if (used_exact_keys) {
        return pairs;
    }

    const std::size_t fallback_count = std::min(left.size(), right.size());
    pairs.reserve(fallback_count);
    for (std::size_t k = 0; k < fallback_count; ++k) {
        pairs.emplace_back(left[k].index, right[k].index);
    }
    return pairs;
}

inline BridgeReport bridge_frames(const TheaterFrame& from, const TheaterFrame& to) {
    BridgeReport report;
    report.from = from.name;
    report.to = to.name;

    const std::size_t count = std::min(from.handles.size(), to.handles.size());
    bool used_exact_keys = false;
    const auto pairs = keyed_bridge_pairs(from, to, used_exact_keys);
    report.compared_pairs = count;
    report.matched_pairs = used_exact_keys ? pairs.size() : 0;
    report.match_coverage = count == 0 ? 0.0L : static_cast<long double>(report.matched_pairs) /
                                             static_cast<long double>(count);
    if (count == 0 || pairs.empty()) {
        report.combined = 1.0L;
        return report;
    }

    long double padic_sum = 0.0L;
    long double theta_sum = 0.0L;
    long double theta_dot = 0.0L;
    long double theta_left_norm = 0.0L;
    long double theta_right_norm = 0.0L;
    long double shell_sum = 0.0L;
    long double frobenioid_sum = 0.0L;
    for (const auto& pair : pairs) {
        const auto left_index = pair.first;
        const auto right_index = pair.second;
        const auto& left = from.handles[left_index];
        const auto& right = to.handles[right_index];
        const long double state_gap = std::abs(left.state - right.state) +
                                      std::abs(left.activation - right.activation);
        padic_sum += adelic_ultrametric_distance(left.prime, right.prime, {2, 3, 5, 7}, 2) *
                     (1.0L + state_gap);

        const long double theta_delta = from.theta_channel[left_index] - to.theta_channel[right_index];
        theta_sum += theta_delta * theta_delta;
        theta_dot += from.theta_channel[left_index] * to.theta_channel[right_index];
        theta_left_norm += from.theta_channel[left_index] * from.theta_channel[left_index];
        theta_right_norm += to.theta_channel[right_index] * to.theta_channel[right_index];

        const long double left_shell = from.xor_indeterminacy[left_index] == 0
                                           ? from.log_shell[left_index]
                                           : -from.log_shell[left_index];
        const long double right_shell = to.xor_indeterminacy[right_index] == 0
                                            ? to.log_shell[right_index]
                                            : -to.log_shell[right_index];
        shell_sum += std::abs(left_shell - right_shell);

        frobenioid_sum += std::abs(from.frobenioid_profile[left_index] -
                                   to.frobenioid_profile[right_index]);
    }

    const long double compared = static_cast<long double>(pairs.size());
    report.padic_distortion = padic_sum / compared;
    report.theta_link_distance = std::sqrt(theta_sum / compared);
    if (theta_left_norm > 0.0L && theta_right_norm > 0.0L) {
        report.theta_correlation = std::clamp(theta_dot / std::sqrt(theta_left_norm * theta_right_norm),
                                              -1.0L,
                                              1.0L);
    }
    report.log_shell_xor_distance = shell_sum / compared;
    report.log_volume_distance = std::abs(from.log_volume - to.log_volume) /
                                 (1.0L + std::max(std::abs(from.log_volume), std::abs(to.log_volume)));
    report.height_gap = std::abs(from.arithmetic_height - to.arithmetic_height) /
                        (1.0L + std::max(std::abs(from.arithmetic_height), std::abs(to.arithmetic_height)));
    report.frobenioid_norm = frobenioid_sum / compared;
    report.determinant_gap = std::abs(from.determinant_log - to.determinant_log) /
                             (1.0L + std::max(std::abs(from.determinant_log), std::abs(to.determinant_log)));
    report.langlands_gap = std::abs(from.langlands_score - to.langlands_score);
    report.zeta_symmetry_gap = std::abs(from.zeta_symmetry_residual - to.zeta_symmetry_residual);
    report.quantum_chaos_gap = std::abs(from.quantum_chaos_score - to.quantum_chaos_score);
    const long double theta_correlation_gap = 0.5L * (1.0L - report.theta_correlation);
    report.combined = 0.15L * report.padic_distortion +
                      0.15L * report.theta_link_distance +
                      0.12L * theta_correlation_gap +
                      0.13L * report.log_shell_xor_distance +
                      0.10L * report.log_volume_distance +
                      0.08L * report.height_gap +
                      0.07L * report.frobenioid_norm +
                      0.04L * report.determinant_gap +
                      0.05L * report.langlands_gap +
                      0.04L * report.zeta_symmetry_gap +
                      0.03L * report.quantum_chaos_gap +
                      0.10L * (1.0L - report.match_coverage);
    return report;
}

inline std::string answer_from_frame(const TheaterFrame& frame, std::string_view prompt) {
    std::vector<Handle> ranked = frame.handles;
    std::sort(ranked.begin(), ranked.end(), [](const Handle& left, const Handle& right) {
        return left.activation > right.activation;
    });

    std::ostringstream out;
    out << "theater=" << frame.name << " phase=" << phase_name(frame.phase) << " primes=";
    const std::size_t count = std::min<std::size_t>(5, ranked.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (i != 0) {
            out << ",";
        }
        out << ranked[i].prime;
    }
    out << " prompt_hash=" << (stable_hash(prompt) % 1'000'000ULL);
    return out.str();
}

inline AnswerInvariant evaluate_invariant(const BridgeReport& ab,
                                          const BridgeReport& ac,
                                          const BridgeReport& bc,
                                          long double tolerance,
                                          long double min_score = 0.5L) {
    const long double max_distortion = std::max({ab.combined, ac.combined, bc.combined});
    AnswerInvariant invariant;
    invariant.score = 1.0L / (1.0L + max_distortion);
    invariant.accepted = max_distortion <= tolerance || invariant.score >= min_score;
    if (max_distortion <= tolerance) {
        invariant.reason = "bridge invariants are inside tolerance";
    } else if (invariant.score >= min_score) {
        invariant.reason = "accepted by bridge score threshold";
    } else {
        invariant.reason = "controlled death: bridge score below threshold";
    }
    return invariant;
}

class IuttEnsemble {
public:
    explicit IuttEnsemble(DzetaConfig config)
        : config_(config),
          cloud_a_("A/main-hodge-theater", config_, 0xA11CE001ULL),
          cloud_b_("B/shadow-hodge-theater", config_, 0xBADC0DE2ULL),
          cloud_c_("C/critic-hodge-theater", config_, 0xC0171C03ULL) {}

    IuttResult resonate(std::string_view prompt, long double tick_time) {
        IuttResult result;
        result.snapshot_a = cloud_a_.tick(prompt, tick_time);

        std::string shadow_prompt(prompt);
        shadow_prompt += " ::theta-link shadow";
        result.snapshot_b = cloud_b_.tick(shadow_prompt, tick_time * 1.031L + 0.001L);

        std::string critic_prompt(prompt);
        critic_prompt += " ::log-shell critic";
        result.snapshot_c = cloud_c_.tick(critic_prompt, tick_time * 0.971L + 0.002L);

        const auto frame_a = make_theater_frame(cloud_a_);
        const auto frame_b = make_theater_frame(cloud_b_);
        const auto frame_c = make_theater_frame(cloud_c_);

        result.bridge_ab = bridge_frames(frame_a, frame_b);
        result.bridge_ac = bridge_frames(frame_a, frame_c);
        result.bridge_bc = bridge_frames(frame_b, frame_c);
        result.invariant = evaluate_invariant(result.bridge_ab, result.bridge_ac, result.bridge_bc,
                                              config_.invariant_tolerance,
                                              config_.invariant_min_score);
        if (config_.math_core_enabled && !result.invariant.accepted) {
            const auto max_clique = std::max({result.snapshot_a.max_clique_found,
                                              result.snapshot_b.max_clique_found,
                                              result.snapshot_c.max_clique_found});
            const auto guaranteed = std::max({result.snapshot_a.guaranteed_clique_r,
                                              result.snapshot_b.guaranteed_clique_r,
                                              result.snapshot_c.guaranteed_clique_r});
            if (max_clique >= config_.min_survival_clique || guaranteed >= config_.min_survival_clique) {
                result.invariant.accepted = true;
                result.invariant.score = std::max(result.invariant.score, config_.invariant_min_score);
                result.invariant.reason = "math core survival floor: coherent clique witness";
            }
        }
        result.accepted = result.invariant.accepted;
        result.answer_a = answer_from_frame(frame_a, prompt);
        result.answer_b = answer_from_frame(frame_b, prompt);
        result.answer_c = answer_from_frame(frame_c, prompt);
        result.seed = result.snapshot_a.answer_seed ^ (result.snapshot_b.answer_seed << 1U) ^
                      (result.snapshot_c.answer_seed << 2U);
        return result;
    }

    Cloud& main_cloud() noexcept {
        return cloud_a_;
    }

    const Cloud& main_cloud() const noexcept {
        return cloud_a_;
    }

private:
    DzetaConfig config_;
    Cloud cloud_a_;
    Cloud cloud_b_;
    Cloud cloud_c_;
};

} // namespace dzeta
