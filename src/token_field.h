#pragma once

#include "code_memory.h"
#include "field_state.h"
#include "zeta_rhythm.h"
#include "zeta_zeros.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dzeta {

using cx = std::complex<long double>;

inline long double padic_norm(long double x, std::uint32_t p) {
    if (std::abs(x) < 1e-30L) return 0.0L;
    long double v = 0, ax = std::abs(x);
    while (ax > 1.0L) { ax /= p; v += 1.0L; }
    while (ax > 0.0L && ax < 1.0L) { ax *= p; v -= 1.0L; }
    return std::pow(static_cast<long double>(p), -v);
}

class OscillatorField {
public:
    explicit OscillatorField(std::size_t max_osc = 4096,
                             std::size_t dim = 192,
                             std::uint64_t seed = 0)
        : max_osc_(std::max<std::size_t>(128, max_osc)),
          dim_(std::max<std::size_t>(16, dim)),
          rng_(seed == 0 ? entropy_seed() : seed) {
        steps_.resize(dim_);
        std::size_t nz = zeta_zero_count();
        for (std::size_t z = 0; z < dim_; ++z) {
            if (z < 64) steps_[z] = 1;           // zeros 0-63, fine
            else if (z < 128) steps_[z] = 4;     // zeros 64-127, every 4th
            else steps_[z] = nz / dim_;           // rest
        }
    }

    bool bad_token(const std::string& t) const {
        if (t.size() > 22) return true;
        if (t.size() <= 1) return true;
        if (t == "\n") return true;
        if (t.find("toolu_") != std::string::npos) return true;
        int digs = 0;
        for (auto c : t) if (c >= '0' && c <= '9') digs++;
        if (digs > (int)t.size() * 2 / 3) return true;
        return false;
    }

    void embed(std::string_view text) {
        auto tokens = tokenize_code(text, std::min<std::size_t>(text.size(), 2048));
        for (auto& t : tokens) {
            if (t.empty() || t == " " || t == "\t") continue;
            if (bad_token(t)) continue;
            if (token_index_.find(t) == token_index_.end()) {
                if (oscs_.size() >= max_osc_) drop_one();
                oscs_.push_back(make_token_oscillator(std::move(t)));
                token_index_[oscs_.back().token] = oscs_.size() - 1U;
            }
        }
    }

    void learn(std::string_view text) {
        auto tokens = tokenize_code(text, std::min<std::size_t>(text.size(), 128));
        if (tokens.size() < 3) { embed(text); return; }
        embed(text);
        std::vector<std::string> context_tokens;
        for (std::size_t ti = 0; ti < tokens.size(); ++ti) {
            if (bad_token(tokens[ti])) {
                continue;
            }
            const std::string context_prefix = context_string(context_tokens);
            const std::string token_prefix = context_prefix.empty() ? tokens[ti] : context_prefix + ' ' + tokens[ti];
            auto found = token_index_.find(tokens[ti]);
            if (found == token_index_.end()) {
                push_context_token(context_tokens, tokens[ti]);
                continue;
            }
            const auto positive_index = found->second;
            if (context_tokens.empty()) {
                push_context_token(context_tokens, tokens[ti]);
                continue;
            }

            auto f_context = make_seed_field_state(context_prefix, 64);
            auto [context_ampl, context_padic] = weyl_transform(f_context);
            auto f_full = make_seed_field_state(token_prefix, 64);
            auto [full_ampl, full_padic] = weyl_transform(f_full);
            if (std::all_of(context_ampl.begin(), context_ampl.end(), [](cx v) { return std::abs(v) < 1e-30L; })) {
                push_context_token(context_tokens, tokens[ti]);
                continue;
            }

            normalize_complex(context_ampl);
            normalize_complex(full_ampl);
            auto transition = spectral_bridge(context_ampl, full_ampl);
            update_oscillator(oscs_[positive_index],
                              context_ampl,
                              full_ampl,
                              transition,
                              context_padic,
                              full_padic,
                              lexical_tail(context_prefix));
            update_contrastive_negatives(positive_index, context_ampl, context_padic);
            push_context_token(context_tokens, tokens[ti]);
        }
    }

    std::string forward(std::string_view text, std::size_t max_tokens = 24) {
        auto field = make_seed_field_state(text, 64);
        auto [fp, current_padic] = weyl_transform(field);
        std::string out;
        std::set<std::string> used;
        auto normalize = [&]() {
            cx n = 0; for (auto v : fp) n += v * std::conj(v);
            long double fn = std::sqrt(std::abs(n));
            if (fn > 1e-30L) for (auto& v : fp) v /= fn;
        };
        normalize();
        const std::vector<cx> prompt_trace = fp;
        const std::vector<long double> prompt_padic_trace = current_padic;
        std::vector<std::uint64_t> active_tail = lexical_tail(text);
        const auto push_active_token = [&](const std::string& token) {
            active_tail.push_back(stable_hash(token));
            constexpr std::size_t tail_limit = 6;
            if (active_tail.size() > tail_limit) {
                active_tail.erase(active_tail.begin(), active_tail.end() - tail_limit);
            }
        };

        for (std::size_t s = 0; s < max_tokens; ++s) {
            for (std::size_t z = 0; z < dim_; ++z) {
                long double theta = static_cast<long double>(s) * 0.005L * (1.0L + static_cast<long double>(z) * 0.1L);
                fp[z] *= cx(std::cos(theta), std::sin(theta));
            }

            struct Candidate {
                long double score;
                std::size_t oscillator;
                std::size_t prototype;
            };
            constexpr std::size_t no_prototype = std::numeric_limits<std::size_t>::max();
            const auto candidate_query = [&](const Candidate& candidate) -> const std::vector<cx>& {
                const auto& oscillator = oscs_[candidate.oscillator];
                if (candidate.prototype != no_prototype && candidate.prototype < oscillator.prototypes.size()) {
                    return oscillator.prototypes[candidate.prototype].query;
                }
                return oscillator.query;
            };
            const auto candidate_key = [&](const Candidate& candidate) -> const std::vector<cx>& {
                const auto& oscillator = oscs_[candidate.oscillator];
                if (candidate.prototype != no_prototype && candidate.prototype < oscillator.prototypes.size()) {
                    return oscillator.prototypes[candidate.prototype].key;
                }
                return oscillator.key;
            };
            const auto candidate_padic = [&](const Candidate& candidate) -> const std::vector<long double>& {
                const auto& oscillator = oscs_[candidate.oscillator];
                if (candidate.prototype != no_prototype && candidate.prototype < oscillator.prototypes.size()) {
                    return oscillator.prototypes[candidate.prototype].padic_signature;
                }
                return oscillator.padic_signature;
            };
            const auto candidate_transition = [&](const Candidate& candidate) -> const std::vector<cx>& {
                const auto& oscillator = oscs_[candidate.oscillator];
                if (candidate.prototype != no_prototype && candidate.prototype < oscillator.prototypes.size()) {
                    return oscillator.prototypes[candidate.prototype].transition;
                }
                return oscillator.transition;
            };
            const auto candidate_next_padic = [&](const Candidate& candidate) -> const std::vector<long double>& {
                const auto& oscillator = oscs_[candidate.oscillator];
                if (candidate.prototype != no_prototype && candidate.prototype < oscillator.prototypes.size()) {
                    return oscillator.prototypes[candidate.prototype].query_padic_signature;
                }
                return oscillator.query_padic_signature;
            };
            const auto candidate_negative_key = [&](const Candidate& candidate) -> const std::vector<cx>& {
                const auto& oscillator = oscs_[candidate.oscillator];
                if (candidate.prototype != no_prototype && candidate.prototype < oscillator.prototypes.size()) {
                    return oscillator.prototypes[candidate.prototype].negative_key;
                }
                return oscillator.negative_key;
            };
            const auto candidate_negative_padic = [&](const Candidate& candidate) -> const std::vector<long double>& {
                const auto& oscillator = oscs_[candidate.oscillator];
                if (candidate.prototype != no_prototype && candidate.prototype < oscillator.prototypes.size()) {
                    return oscillator.prototypes[candidate.prototype].negative_padic_signature;
                }
                return oscillator.negative_padic_signature;
            };
            const auto candidate_tail = [&](const Candidate& candidate) -> const std::vector<std::uint64_t>& {
                const auto& oscillator = oscs_[candidate.oscillator];
                if (candidate.prototype != no_prototype && candidate.prototype < oscillator.prototypes.size()) {
                    return oscillator.prototypes[candidate.prototype].context_tail;
                }
                static const std::vector<std::uint64_t> empty_tail;
                return empty_tail;
            };

            // compute raw delta matches
            std::vector<Candidate> raw;
            for (std::size_t i = 0; i < oscs_.size(); ++i) {
                if (used.find(oscs_[i].token) != used.end()) continue;
                if (oscs_[i].token.size() <= 1) continue;
                const std::size_t prototypes = std::max<std::size_t>(1, oscs_[i].prototypes.size());
                for (std::size_t p = 0; p < prototypes; ++p) {
                    const Candidate candidate{0.0L, i, oscs_[i].prototypes.empty() ? no_prototype : p};
                    cx dm = 0;
                    const auto& key = candidate_key(candidate);
                    for (std::size_t j = 0; j < dim_; ++j) {
                        dm += std::conj(key[j]) * fp[j];
                    }
                    const long double padic_match = 0.5L + 0.5L * cosine(current_padic, candidate_padic(candidate));
                    const long double observations =
                        static_cast<long double>(std::max<std::size_t>(1, oscs_[i].observations));
                    const long double frequency_penalty = 1.0L / std::sqrt(1.0L + 0.03L * observations);
                    const long double content_gain =
                        std::clamp(static_cast<long double>(oscs_[i].token.size()) / 7.0L, 0.55L, 1.35L);
                    const long double lexical_match = tail_overlap(active_tail, candidate_tail(candidate));
                    const long double context_gate = 0.40L + 0.90L * lexical_match;
                    const auto bridged = apply_bridge(fp, candidate_transition(candidate));
                    const long double bridge_fit = complex_similarity(bridged, candidate_query(candidate));
                    const long double negative_spectral = complex_similarity(fp, candidate_negative_key(candidate));
                    const long double negative_padic =
                        0.5L + 0.5L * cosine(current_padic, candidate_negative_padic(candidate));
                    const long double collision =
                        std::clamp(0.78L * negative_spectral + 0.22L * negative_padic, 0.0L, 1.0L);
                    const long double contrastive_penalty =
                        std::clamp(1.0L - contrastive_strength_ * collision, 0.12L, 1.0L);
                    const long double reliability =
                        oscs_[i].strength * frequency_penalty * content_gain / (1.0L + oscs_[i].error_ema);
                    long double score =
                        std::abs(dm) * (0.70L + 0.30L * padic_match) *
                        (0.75L + 0.25L * bridge_fit) * reliability * context_gate * contrastive_penalty;
                    if (score > 1e-12L) {
                        raw.push_back({score, i, candidate.prototype});
                    }
                }
            }
            if (raw.empty()) break;
            const auto by_score = [](const Candidate& left, const Candidate& right) {
                return left.score > right.score;
            };
            std::partial_sort(raw.begin(), raw.begin() + std::min<std::size_t>(64, raw.size()),
                              raw.end(), by_score);
            std::size_t K = std::min<std::size_t>(64, raw.size());
            // lateral inhibition: suppress similar oscillators
            std::vector<Candidate> inhibited;
            for (std::size_t t = 0; t < K; ++t) {
                Candidate candidate = raw[t];
                long double score = candidate.score;
                const auto& q = candidate_query(candidate);
                for (std::size_t u = 0; u < t; ++u) {
                    const auto& qu = candidate_query(raw[u]);
                    cx cross = 0;
                    long double n1 = 0, n2 = 0;
                    for (std::size_t j = 0; j < dim_; ++j) {
                        cross += std::conj(q[j]) * qu[j];
                        n1 += std::abs(q[j]) * std::abs(q[j]);
                        n2 += std::abs(qu[j]) * std::abs(qu[j]);
                    }
                    long double sim = std::abs(cross) / (std::sqrt(n1 * n2) + 1e-30L);
                    score -= 0.3L * sim * candidate.score;  // inhibition by similarity
                }
                candidate.score = score;
                inhibited.push_back(candidate);
            }
            if (generation_temperature_ > 0.0L && inhibited.size() > 1) {
                const std::size_t stochastic_k = std::min<std::size_t>(8, inhibited.size());
                for (std::size_t i = 0; i < stochastic_k; ++i) {
                    const long double scale = std::max<long double>(1.0e-12L, std::abs(inhibited[i].score));
                    inhibited[i].score += generation_temperature_ * scale * centered_noise();
                }
            }
            std::partial_sort(inhibited.begin(), inhibited.begin() + 1, inhibited.end(), by_score);
            const Candidate best = inhibited[0];
            std::size_t best_i = best.oscillator;
            if (!out.empty()) out += ' ';
            out += oscs_[best_i].token;
            used.insert(oscs_[best_i].token);
            push_active_token(oscs_[best_i].token);
            const auto bridged = apply_bridge(fp, candidate_transition(best));
            const auto& query = candidate_query(best);
            for (std::size_t j = 0; j < dim_; ++j) {
                const long double trace = 0.22L / (1.0L + 0.18L * static_cast<long double>(s));
                const long double query_weight = 0.58L;
                const long double bridge_weight = std::max(0.0L, 1.0L - query_weight - trace);
                fp[j] = query_weight * query[j] + bridge_weight * bridged[j] + trace * prompt_trace[j];
            }
            const auto& next_padic = candidate_next_padic(best);
            for (std::size_t j = 0; j < std::min(current_padic.size(), next_padic.size()); ++j) {
                const long double trace = 0.18L / (1.0L + 0.20L * static_cast<long double>(s));
                current_padic[j] = (1.0L - trace) * next_padic[j] + trace * prompt_padic_trace[j];
            }
            normalize_real(current_padic);
            normalize();
        }
        return out;
    }

    std::size_t size() const noexcept { return oscs_.size(); }
    void clear() {
        oscs_.clear();
        token_index_.clear();
        total_observations_ = 0;
        contrastive_updates_ = 0;
        loss_updates_ = 0;
        loss_ema_ = 0.0L;
    }
    std::size_t observation_count() const noexcept { return total_observations_; }
    std::size_t contrastive_update_count() const noexcept { return contrastive_updates_; }
    long double mean_loss() const noexcept { return loss_updates_ == 0 ? 0.0L : loss_ema_; }

    void set_generation_temperature(long double temperature) noexcept {
        generation_temperature_ = std::clamp(temperature, 0.0L, 2.0L);
    }

    void set_learning_rate(long double rate) noexcept {
        learning_rate_ = std::clamp(rate, 1.0e-4L, 1.0L);
    }

private:
    struct ContextPrototype {
        std::vector<cx> key;
        std::vector<cx> query;
        std::vector<cx> transition;
        std::vector<long double> padic_signature;
        std::vector<long double> query_padic_signature;
        std::vector<cx> negative_key;
        std::vector<long double> negative_padic_signature;
        std::vector<std::uint64_t> context_tail;
        long double error_ema = 1.0L;
        std::size_t observations = 0;
    };

    struct TokenOscillator {
        std::string token;
        std::vector<cx> query;    // next-state projection (where to go)
        std::vector<cx> key;      // current-state projection (where we are)
        std::vector<cx> transition; // learned phase bridge from key to query
        std::vector<long double> padic_signature;
        std::vector<long double> query_padic_signature;
        std::vector<cx> negative_key;
        std::vector<long double> negative_padic_signature;
        long double strength = 1.0L;
        long double error_ema = 1.0L;
        std::size_t observations = 0;
        std::vector<ContextPrototype> prototypes;
    };

    TokenOscillator make_token_oscillator(std::string token) const {
        return {std::move(token),
                std::vector<cx>(dim_, cx(0, 0)),
                std::vector<cx>(dim_, cx(0, 0)),
                std::vector<cx>(dim_, cx(1, 0)),
                std::vector<long double>(dim_, 0.0L),
                std::vector<long double>(dim_, 0.0L),
                std::vector<cx>(dim_, cx(0, 0)),
                std::vector<long double>(dim_, 0.0L),
                1.0L,
                1.0L,
                0,
                {}};
    }

    void push_context_token(std::vector<std::string>& context_tokens,
                            const std::string& token) const {
        context_tokens.push_back(token);
        if (context_tokens.size() > max_context_tokens_) {
            context_tokens.erase(context_tokens.begin(),
                                 context_tokens.begin() + static_cast<std::ptrdiff_t>(context_tokens.size() -
                                                                                       max_context_tokens_));
        }
    }

    static std::string context_string(const std::vector<std::string>& context_tokens) {
        std::string context;
        for (const auto& token : context_tokens) {
            if (!context.empty()) {
                context += ' ';
            }
            context += token;
        }
        return context;
    }

    static std::uint64_t entropy_seed() {
        std::random_device rd;
        const auto now = static_cast<std::uint64_t>(std::time(nullptr));
        return (static_cast<std::uint64_t>(rd()) << 32U) ^ static_cast<std::uint64_t>(rd()) ^
               (now * 0x9e3779b97f4a7c15ULL);
    }

    static void normalize_complex(std::vector<cx>& values) {
        long double norm = 0.0L;
        for (auto value : values) {
            norm += std::norm(value);
        }
        if (norm <= 1.0e-30L) {
            return;
        }
        norm = std::sqrt(norm);
        for (auto& value : values) {
            value /= norm;
        }
    }

    static long double complex_loss(const std::vector<cx>& left, const std::vector<cx>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0L;
        }
        long double loss = 0.0L;
        for (std::size_t i = 0; i < count; ++i) {
            loss += std::norm(left[i] - right[i]);
        }
        return loss / static_cast<long double>(count);
    }

    static long double complex_similarity(const std::vector<cx>& left, const std::vector<cx>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0L;
        }
        cx dot = 0;
        long double left_norm = 0.0L;
        long double right_norm = 0.0L;
        for (std::size_t i = 0; i < count; ++i) {
            dot += std::conj(left[i]) * right[i];
            left_norm += std::norm(left[i]);
            right_norm += std::norm(right[i]);
        }
        if (left_norm <= 1.0e-30L || right_norm <= 1.0e-30L) {
            return 0.0L;
        }
        return std::clamp(std::abs(dot) / std::sqrt(left_norm * right_norm), 0.0L, 1.0L);
    }

    static std::vector<cx> spectral_bridge(const std::vector<cx>& from, const std::vector<cx>& to) {
        const std::size_t count = std::min(from.size(), to.size());
        std::vector<cx> bridge(count, cx(1, 0));
        for (std::size_t i = 0; i < count; ++i) {
            const long double magnitude = std::max<long double>(1.0e-12L, std::abs(from[i]));
            bridge[i] = to[i] * std::conj(from[i]) / magnitude;
        }
        normalize_complex(bridge);
        return bridge;
    }

    static std::vector<cx> apply_bridge(const std::vector<cx>& state, const std::vector<cx>& bridge) {
        const std::size_t count = std::min(state.size(), bridge.size());
        std::vector<cx> projected(state.size(), cx(0, 0));
        for (std::size_t i = 0; i < count; ++i) {
            projected[i] = state[i] * bridge[i];
        }
        normalize_complex(projected);
        return projected;
    }

    static void mix_negative_key(std::vector<cx>& negative_key,
                                 const std::vector<cx>& target_key,
                                 long double rate) {
        const std::size_t count = std::min(negative_key.size(), target_key.size());
        for (std::size_t i = 0; i < count; ++i) {
            negative_key[i] = (1.0L - rate) * negative_key[i] + rate * target_key[i];
        }
        normalize_complex(negative_key);
    }

    static void mix_negative_padic(std::vector<long double>& negative_padic,
                                   const std::vector<long double>& target_padic,
                                   long double rate) {
        const std::size_t count = std::min(negative_padic.size(), target_padic.size());
        for (std::size_t i = 0; i < count; ++i) {
            negative_padic[i] = (1.0L - rate) * negative_padic[i] + rate * target_padic[i];
        }
        normalize_real(negative_padic);
    }

    static void repel_from(std::vector<cx>& vector,
                           const std::vector<cx>& away,
                           long double rate) {
        const std::size_t count = std::min(vector.size(), away.size());
        if (count == 0) {
            return;
        }
        cx projection = 0;
        for (std::size_t i = 0; i < count; ++i) {
            projection += std::conj(away[i]) * vector[i];
        }
        for (std::size_t i = 0; i < count; ++i) {
            vector[i] -= rate * projection * away[i];
        }
        normalize_complex(vector);
    }

    static long double cosine(const std::vector<long double>& left, const std::vector<long double>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0L;
        }
        long double dot = 0.0L;
        long double left_norm = 0.0L;
        long double right_norm = 0.0L;
        for (std::size_t i = 0; i < count; ++i) {
            dot += left[i] * right[i];
            left_norm += left[i] * left[i];
            right_norm += right[i] * right[i];
        }
        if (left_norm <= 1.0e-30L || right_norm <= 1.0e-30L) {
            return 0.0L;
        }
        return std::clamp(dot / std::sqrt(left_norm * right_norm), -1.0L, 1.0L);
    }

    static void normalize_real(std::vector<long double>& values) {
        long double norm = 0.0L;
        for (auto value : values) {
            norm += value * value;
        }
        if (norm <= 1.0e-30L) {
            return;
        }
        norm = std::sqrt(norm);
        for (auto& value : values) {
            value /= norm;
        }
    }

    static std::vector<std::uint64_t> lexical_tail(std::string_view text) {
        constexpr std::size_t tail_limit = 6;
        std::vector<std::uint64_t> tail;
        for (const auto& token : tokenize_query(text)) {
            if (token.size() <= 1) {
                continue;
            }
            tail.push_back(stable_hash(token));
            if (tail.size() > tail_limit) {
                tail.erase(tail.begin());
            }
        }
        return tail;
    }

    static long double tail_overlap(const std::vector<std::uint64_t>& active_tail,
                                    const std::vector<std::uint64_t>& prototype_tail) {
        if (active_tail.empty() || prototype_tail.empty()) {
            return 0.0L;
        }
        long double score = 0.0L;
        long double denom = 0.0L;
        for (std::size_t ai = 0; ai < active_tail.size(); ++ai) {
            const std::size_t active_recency = active_tail.size() - 1U - ai;
            const long double base_weight = 1.0L / (1.0L + static_cast<long double>(active_recency));
            denom += base_weight;
            long double best = 0.0L;
            for (std::size_t pi = 0; pi < prototype_tail.size(); ++pi) {
                if (active_tail[ai] != prototype_tail[pi]) {
                    continue;
                }
                const std::size_t proto_recency = prototype_tail.size() - 1U - pi;
                const auto distance = active_recency > proto_recency
                                          ? active_recency - proto_recency
                                          : proto_recency - active_recency;
                best = std::max(best,
                                base_weight / (1.0L + 0.5L * static_cast<long double>(distance)));
            }
            score += best;
        }
        return denom <= 1.0e-30L ? 0.0L : std::clamp(score / denom, 0.0L, 1.0L);
    }

    long double random_unit() const {
        return std::generate_canonical<long double, std::numeric_limits<long double>::digits>(rng_);
    }

    long double centered_noise() const {
        return 2.0L * random_unit() - 1.0L;
    }

    void update_oscillator(TokenOscillator& oscillator,
                           const std::vector<cx>& target_key,
                           const std::vector<cx>& target_query,
                           const std::vector<cx>& target_transition,
                           const std::vector<long double>& target_padic,
                           const std::vector<long double>& target_query_padic,
                           const std::vector<std::uint64_t>& context_tail) {
        const long double before_loss =
            0.5L * complex_loss(oscillator.key, target_key) +
            0.5L * complex_loss(oscillator.query, target_query);
        const long double rate = oscillator.observations == 0
                                     ? 1.0L
                                     : learning_rate_ / std::sqrt(1.0L + 0.02L * static_cast<long double>(oscillator.observations));
        for (std::size_t j = 0; j < dim_; ++j) {
            oscillator.key[j] = (1.0L - rate) * oscillator.key[j] + rate * target_key[j];
            oscillator.query[j] = (1.0L - rate) * oscillator.query[j] + rate * target_query[j];
            oscillator.transition[j] = (1.0L - rate) * oscillator.transition[j] + rate * target_transition[j];
            oscillator.padic_signature[j] =
                (1.0L - rate) * oscillator.padic_signature[j] + rate * target_padic[j];
            oscillator.query_padic_signature[j] =
                (1.0L - rate) * oscillator.query_padic_signature[j] + rate * target_query_padic[j];
        }
        normalize_complex(oscillator.key);
        normalize_complex(oscillator.query);
        normalize_complex(oscillator.transition);
        normalize_real(oscillator.padic_signature);
        normalize_real(oscillator.query_padic_signature);

        std::size_t best_prototype = std::numeric_limits<std::size_t>::max();
        long double best_match = -1.0L;
        for (std::size_t i = 0; i < oscillator.prototypes.size(); ++i) {
            const auto& prototype = oscillator.prototypes[i];
            const long double spectral_match = complex_similarity(prototype.key, target_key);
            const long double padic_match = 0.5L + 0.5L * cosine(prototype.padic_signature, target_padic);
            const long double match = 0.82L * spectral_match + 0.18L * padic_match;
            if (match > best_match) {
                best_match = match;
                best_prototype = i;
            }
        }

        if (best_prototype == std::numeric_limits<std::size_t>::max() ||
            (best_match < 0.74L && oscillator.prototypes.size() < max_prototypes_per_token_)) {
            oscillator.prototypes.push_back({target_key,
                                             target_query,
                                             target_transition,
                                             target_padic,
                                             target_query_padic,
                                             std::vector<cx>(dim_, cx(0, 0)),
                                             std::vector<long double>(dim_, 0.0L),
                                             context_tail,
                                             before_loss,
                                             1});
        } else {
            auto& prototype = oscillator.prototypes[best_prototype];
            const long double proto_loss =
                0.5L * complex_loss(prototype.key, target_key) +
                0.5L * complex_loss(prototype.query, target_query);
            const long double proto_rate =
                prototype.observations == 0
                    ? 1.0L
                    : std::min<long double>(0.65L, rate * 1.25L);
            for (std::size_t j = 0; j < dim_; ++j) {
                prototype.key[j] = (1.0L - proto_rate) * prototype.key[j] + proto_rate * target_key[j];
                prototype.query[j] = (1.0L - proto_rate) * prototype.query[j] + proto_rate * target_query[j];
                prototype.transition[j] =
                    (1.0L - proto_rate) * prototype.transition[j] + proto_rate * target_transition[j];
                prototype.padic_signature[j] =
                    (1.0L - proto_rate) * prototype.padic_signature[j] + proto_rate * target_padic[j];
                prototype.query_padic_signature[j] =
                    (1.0L - proto_rate) * prototype.query_padic_signature[j] + proto_rate * target_query_padic[j];
            }
            normalize_complex(prototype.key);
            normalize_complex(prototype.query);
            normalize_complex(prototype.transition);
            normalize_real(prototype.padic_signature);
            normalize_real(prototype.query_padic_signature);
            ++prototype.observations;
            prototype.error_ema = 0.90L * prototype.error_ema + 0.10L * proto_loss;
        }

        ++oscillator.observations;
        ++total_observations_;
        oscillator.error_ema = oscillator.observations == 1
                                   ? before_loss
                                   : 0.92L * oscillator.error_ema + 0.08L * before_loss;
        oscillator.strength = std::clamp(0.35L + std::log1p(static_cast<long double>(oscillator.observations)) /
                                                    (1.0L + oscillator.error_ema),
                                         0.10L,
                                         8.0L);
        loss_ema_ = loss_updates_ == 0 ? before_loss : 0.98L * loss_ema_ + 0.02L * before_loss;
        ++loss_updates_;
    }

    void update_contrastive_negatives(std::size_t positive_index,
                                      const std::vector<cx>& target_key,
                                      const std::vector<long double>& target_padic) {
        if (oscs_.size() < 2) {
            return;
        }
        if (contrastive_period_ > 1 && total_observations_ % contrastive_period_ != 0) {
            return;
        }

        struct HardNegative {
            long double score;
            std::size_t oscillator;
            std::size_t prototype;
        };

        constexpr std::size_t no_prototype = std::numeric_limits<std::size_t>::max();
        std::vector<HardNegative> candidates;
        candidates.reserve(std::min<std::size_t>(oscs_.size(), 256));
        for (std::size_t i = 0; i < oscs_.size(); ++i) {
            if (i == positive_index || oscs_[i].observations == 0) {
                continue;
            }
            const std::size_t prototypes = std::max<std::size_t>(1, oscs_[i].prototypes.size());
            for (std::size_t p = 0; p < prototypes; ++p) {
                const bool has_prototype = !oscs_[i].prototypes.empty();
                const auto& key = has_prototype ? oscs_[i].prototypes[p].key : oscs_[i].key;
                const auto& padic = has_prototype ? oscs_[i].prototypes[p].padic_signature : oscs_[i].padic_signature;
                const long double spectral_match = complex_similarity(key, target_key);
                const long double padic_match = 0.5L + 0.5L * cosine(padic, target_padic);
                const long double score = 0.84L * spectral_match + 0.16L * padic_match;
                if (score > contrastive_margin_) {
                    candidates.push_back({score, i, has_prototype ? p : no_prototype});
                }
            }
        }
        if (candidates.empty()) {
            return;
        }

        const auto by_score = [](const HardNegative& left, const HardNegative& right) {
            return left.score > right.score;
        };
        const std::size_t take = std::min(max_hard_negatives_, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end(), by_score);
        for (std::size_t idx = 0; idx < take; ++idx) {
            auto& oscillator = oscs_[candidates[idx].oscillator];
            const long double rate = std::clamp(contrastive_rate_ * candidates[idx].score, 0.005L, 0.14L);
            mix_negative_key(oscillator.negative_key, target_key, rate);
            mix_negative_padic(oscillator.negative_padic_signature, target_padic, rate);
            repel_from(oscillator.key, target_key, rate * 0.35L);
            if (candidates[idx].prototype != no_prototype &&
                candidates[idx].prototype < oscillator.prototypes.size()) {
                auto& prototype = oscillator.prototypes[candidates[idx].prototype];
                mix_negative_key(prototype.negative_key, target_key, rate);
                mix_negative_padic(prototype.negative_padic_signature, target_padic, rate);
                repel_from(prototype.key, target_key, rate * 0.45L);
            }
            ++contrastive_updates_;
        }
    }

    std::pair<std::vector<cx>, std::vector<long double>> weyl_transform(const FieldState& f) const {
        std::vector<cx> ampl(dim_, cx(0, 0));
        std::vector<long double> padic(dim_, 0.0L);
        if (f.empty()) return {ampl, padic};
        for (std::size_t i = 0; i < std::min<std::size_t>(f.size(), 256); ++i) {
            long double act = f.activations[i];
            long double en = f.energy[i];
            long double theta = f.theta[i];
            long double charge = f.semantic_charge[i];
            for (std::size_t z = 0; z < dim_; ++z) {
                long double zr = zeta_zero((z * steps_[z]) % zeta_zero_count());
                long double phase_dither =
                    0.03L * std::sin(charge * (static_cast<long double>(z) + 1.0L) * 12.9898L +
                                      static_cast<long double>(f.primes[i]) * 0.0174533L);
                long double re = std::cos(theta * zr + charge * 0.5L + phase_dither);
                long double im = std::sin(theta * zr + charge * 0.5L + phase_dither);
                ampl[z] += cx(act * en * re, act * en * im);
                padic[z] += act * en * padic_norm(f.padic_coordinates[i], f.primes[i % f.size()] % 997U + 2U);
            }
        }
        long double an = 0;
        for (auto v : ampl) an += std::abs(v) * std::abs(v);
        if (an > 1e-30L) { an = std::sqrt(an); for (auto& v : ampl) v /= an; }
        long double pn = 0;
        for (auto v : padic) pn += v * v;
        if (pn > 1e-30L) { pn = std::sqrt(pn); for (auto& v : padic) v /= pn; }
        return {ampl, padic};
    }

    static void complex_perturb(FieldState& f, const std::vector<cx>& coupling) {
        for (std::size_t i = 0; i < std::min<std::size_t>(f.size(), coupling.size()); ++i) {
            f.phases[i] = wrap_phase(f.phases[i] + std::arg(coupling[i]) * 0.15L);
            f.activations[i] = std::clamp(f.activations[i] + 0.03L * std::abs(coupling[i]), 0.0L, 1.0L);
        }
    }

    void drop_one() {
        std::size_t wi = 0; long double ws = 1e18;
        for (std::size_t i = 0; i < oscs_.size(); ++i) {
            long double s = oscs_[i].observations;
            if (s < ws) { ws = s; wi = i; }
        }
        token_index_.erase(oscs_[wi].token);
        if (wi + 1 < oscs_.size()) oscs_[wi] = std::move(oscs_.back());
        if (wi < oscs_.size() - 1U) {
            token_index_[oscs_[wi].token] = wi;
        }
        oscs_.pop_back();
    }

    std::vector<TokenOscillator> oscs_;
    std::unordered_map<std::string, std::size_t> token_index_;
    std::vector<cx> fp_cx_;
    std::size_t max_osc_;
    std::size_t dim_;
    std::vector<std::size_t> steps_;
    mutable std::mt19937_64 rng_;
    long double learning_rate_ = 0.32L;
    long double generation_temperature_ = 0.08L;
    long double contrastive_rate_ = 0.08L;
    long double contrastive_margin_ = 0.62L;
    long double contrastive_strength_ = 0.74L;
    std::size_t max_prototypes_per_token_ = 4;
    std::size_t max_hard_negatives_ = 4;
    std::size_t max_context_tokens_ = 24;
    std::size_t contrastive_period_ = 3;
    std::size_t total_observations_ = 0;
    std::size_t contrastive_updates_ = 0;
    std::size_t loss_updates_ = 0;
    long double loss_ema_ = 0.0L;
};

} // namespace dzeta
