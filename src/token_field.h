#pragma once

#include "code_memory.h"
#include "field_state.h"
#include "zeta_rhythm.h"
#include "zeta_zeros.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <istream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <condition_variable>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

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
    struct TokenSummary {
        std::string token;
        std::size_t observations = 0;
        std::size_t prototypes = 0;
        long double strength = 0.0L;
        long double error_ema = 0.0L;
    };

    struct TokenLink {
        std::string token;
        std::size_t observations = 0;
        long double association_score = 0.0L;
        long double next_similarity = 0.0L;
        long double context_similarity = 0.0L;
        long double transition_similarity = 0.0L;
        long double padic_similarity = 0.0L;
    };

    class RangeThreadPool {
    public:
        explicit RangeThreadPool(std::size_t max_workers)
            : max_workers_(std::max<std::size_t>(1, max_workers)) {
            threads_.reserve(max_workers_ > 0 ? max_workers_ - 1U : 0U);
            for (std::size_t i = 0; i + 1U < max_workers_; ++i) {
                threads_.emplace_back([this, i]() { worker_loop(i); });
            }
        }

        RangeThreadPool(const RangeThreadPool&) = delete;
        RangeThreadPool& operator=(const RangeThreadPool&) = delete;

        ~RangeThreadPool() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
                ++generation_;
            }
            work_cv_.notify_all();
            for (auto& thread : threads_) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }

        std::size_t max_workers() const noexcept { return max_workers_; }

        template <typename Fn>
        void run(std::size_t work_items, std::size_t workers, Fn&& fn) {
            workers = std::min({workers, max_workers_, work_items});
            if (workers <= 1) {
                fn(0, work_items);
                return;
            }

            std::vector<std::pair<std::size_t, std::size_t>> local_ranges;
            local_ranges.reserve(workers);
            const std::size_t block = (work_items + workers - 1U) / workers;
            std::size_t begin = 0;
            while (begin < work_items && local_ranges.size() < workers) {
                const std::size_t end = std::min(work_items, begin + block);
                local_ranges.emplace_back(begin, end);
                begin = end;
            }
            workers = local_ranges.size();
            if (workers <= 1) {
                fn(0, work_items);
                return;
            }

            using TaskFn = std::decay_t<Fn>;
            auto task_holder = std::make_shared<TaskFn>(std::forward<Fn>(fn));
            std::pair<std::size_t, std::size_t> main_range{0, 0};
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ranges_ = std::move(local_ranges);
                task_ = [task_holder](std::size_t range_begin, std::size_t range_end) {
                    (*task_holder)(range_begin, range_end);
                };
                active_workers_ = workers - 1U;
                remaining_workers_ = active_workers_;
                main_range = ranges_[workers - 1U];
                ++generation_;
            }
            work_cv_.notify_all();

            (*task_holder)(main_range.first, main_range.second);

            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [&]() { return remaining_workers_ == 0; });
            task_ = {};
            active_workers_ = 0;
        }

    private:
        void worker_loop(std::size_t worker_index) {
            std::size_t seen_generation = 0;
            while (true) {
                std::function<void(std::size_t, std::size_t)> task;
                std::pair<std::size_t, std::size_t> range{0, 0};
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    work_cv_.wait(lock, [&]() {
                        return stop_ || generation_ != seen_generation;
                    });
                    if (stop_) {
                        return;
                    }
                    seen_generation = generation_;
                    if (worker_index >= active_workers_) {
                        continue;
                    }
                    range = ranges_[worker_index];
                    task = task_;
                }

                task(range.first, range.second);

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    --remaining_workers_;
                    if (remaining_workers_ == 0) {
                        done_cv_.notify_one();
                    }
                }
            }
        }

        std::size_t max_workers_;
        std::vector<std::thread> threads_;
        std::mutex mutex_;
        std::condition_variable work_cv_;
        std::condition_variable done_cv_;
        std::vector<std::pair<std::size_t, std::size_t>> ranges_;
        std::function<void(std::size_t, std::size_t)> task_;
        std::size_t active_workers_ = 0;
        std::size_t remaining_workers_ = 0;
        std::size_t generation_ = 0;
        bool stop_ = false;
    };

    explicit OscillatorField(std::size_t max_osc = 4096,
                             std::size_t dim = 192,
                             std::uint64_t seed = 0)
        : max_osc_(std::max<std::size_t>(128, max_osc)),
          dim_(std::max<std::size_t>(16, dim)),
          thread_count_(default_thread_count()),
          parallel_min_dimensions_(default_parallel_min_dimensions()),
          rng_(seed == 0 ? entropy_seed() : seed) {
        initialize_spectral_basis();
        initialize_seed_projection(64);
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
        embed_tokens(tokens);
    }

    void embed_tokens(const std::vector<std::string>& tokens) {
        for (const auto& t : tokens) {
            if (t.empty() || t == " " || t == "\t") continue;
            if (bad_token(t)) continue;
            if (token_index_.find(t) == token_index_.end()) {
                if (oscs_.size() >= max_osc_) drop_one();
                oscs_.push_back(make_token_oscillator(t));
                token_index_[oscs_.back().token] = oscs_.size() - 1U;
            }
        }
    }

    void learn(std::string_view text) {
        auto tokens = tokenize_code(text, std::min<std::size_t>(text.size(), 2048));
        if (tokens.size() < 3) { embed_tokens(tokens); return; }
        embed_tokens(tokens);
        const std::size_t train_tokens = std::min<std::size_t>(tokens.size(), 128);
        struct FieldProjection {
            std::string text;
            std::vector<cx> ampl;
            std::vector<long double> padic;
            bool valid = false;
        };
        auto fill_projection = [&](FieldProjection& projection, std::string projection_text) {
            projection.text = std::move(projection_text);
            seed_weyl_transform_into(projection.text, projection.ampl, projection.padic, false);
            projection.valid = true;
        };
        FieldProjection previous_projection;
        FieldProjection current_projection;
        FieldProjection context_scratch;
        std::vector<cx> transition;
        std::vector<std::string> context_tokens;
        for (std::size_t ti = 0; ti < train_tokens; ++ti) {
            if (bad_token(tokens[ti])) {
                continue;
            }
            const std::string context_prefix = context_string(context_tokens);
            const std::string token_prefix = context_prefix.empty() ? tokens[ti] : context_prefix + ' ' + tokens[ti];
            const bool projection_remains_context = context_tokens.size() + 1U <= max_context_tokens_;
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

            FieldProjection* context_projection = nullptr;
            if (previous_projection.valid && previous_projection.text == context_prefix) {
                context_projection = &previous_projection;
            } else {
                fill_projection(context_scratch, context_prefix);
                context_projection = &context_scratch;
            }
            fill_projection(current_projection, token_prefix);
            if (std::all_of(context_projection->ampl.begin(), context_projection->ampl.end(), [](cx v) {
                    return std::abs(v) < 1e-30L;
                })) {
                push_context_token(context_tokens, tokens[ti]);
                previous_projection.valid = false;
                continue;
            }

            if (update_probability_ < 1.0L && random_unit() > update_probability_) {
                push_context_token(context_tokens, tokens[ti]);
                if (projection_remains_context) {
                    std::swap(previous_projection, current_projection);
                } else {
                    previous_projection.valid = false;
                }
                continue;
            }
            if (update_noise_ > 0.0L) {
                add_complex_noise(context_projection->ampl, update_noise_);
                add_complex_noise(current_projection.ampl, update_noise_);
                add_real_noise(context_projection->padic, update_noise_);
                add_real_noise(current_projection.padic, update_noise_);
            }
            spectral_bridge_into(context_projection->ampl, current_projection.ampl, transition);
            update_oscillator(oscs_[positive_index],
                              context_projection->ampl,
                              current_projection.ampl,
                              transition,
                              context_projection->padic,
                              current_projection.padic,
                              lexical_tail(context_prefix));
            update_contrastive_negatives(positive_index, context_projection->ampl, context_projection->padic);
            push_context_token(context_tokens, tokens[ti]);
            if (projection_remains_context) {
                std::swap(previous_projection, current_projection);
            } else {
                previous_projection.valid = false;
            }
        }
    }

    std::string forward(std::string_view text, std::size_t max_tokens = 24) {
        auto [fp, current_padic] = seed_weyl_transform(text, false);
        std::string out;
        std::set<std::string> used;
        auto normalize = [&]() {
            long double n = 0.0L;
            for (auto v : fp) n += complex_norm(v);
            long double fn = std::sqrt(n);
            if (fn > 1e-30L) for (auto& v : fp) v /= fn;
        };
        normalize();
        inject_prompt_resonance(text, fp, current_padic);
        normalize();
        const std::vector<cx> prompt_trace = fp;
        const std::vector<long double> prompt_padic_trace = current_padic;
        std::vector<cx> attractor_center;
        std::vector<long double> attractor_padic_center;
        if (dimension_interference_ > 0.0L) {
            build_attractor_center(attractor_center, attractor_padic_center);
        }
        std::vector<cx> prompt_delta = prompt_trace;
        std::vector<long double> prompt_padic_delta = prompt_padic_trace;
        if (dimension_interference_ > 0.0L) {
            remove_attractor_projection(prompt_delta, attractor_center);
            remove_attractor_projection(prompt_padic_delta, attractor_padic_center);
        }
        struct PromptAxis {
            std::vector<cx> field;
            std::vector<long double> padic;
            long double weight = 1.0L;
        };
        std::vector<PromptAxis> prompt_axes;
        if (dimension_interference_ > 0.0L) {
            const auto prompt_tokens = tokenize_query(text);
            prompt_axes.reserve(prompt_tokens.size() * 2U + 1U);
            const auto add_prompt_axis = [&](std::string_view axis_text, long double weight) {
                PromptAxis axis;
                seed_weyl_transform_into(axis_text, axis.field, axis.padic, true, false);
                remove_attractor_projection(axis.field, attractor_center);
                remove_attractor_projection(axis.padic, attractor_padic_center);
                axis.weight = weight;
                prompt_axes.push_back(std::move(axis));
            };
            for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
                if (prompt_tokens[i].size() > 1) {
                    add_prompt_axis(prompt_tokens[i],
                                    std::clamp(static_cast<long double>(prompt_tokens[i].size()) / 6.0L,
                                               0.65L,
                                               1.65L));
                }
                if (i > 0 && prompt_tokens[i - 1].size() > 1 && prompt_tokens[i].size() > 1) {
                    add_prompt_axis(prompt_tokens[i - 1] + " " + prompt_tokens[i], 1.35L);
                }
            }
            add_prompt_axis(text, 1.0L);
        }
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
                long double st = 0.0L;
                long double ct = 1.0L;
                sincos_ld(theta, st, ct);
                fp[z] *= cx(ct, st);
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
                if (is_subword_continuation(oscs_[i].token)) continue;
                const std::size_t prototypes = std::max<std::size_t>(1, oscs_[i].prototypes.size());
                for (std::size_t p = 0; p < prototypes; ++p) {
                    const Candidate candidate{0.0L, i, oscs_[i].prototypes.empty() ? no_prototype : p};
                    cx dm = 0;
                    const auto& key = candidate_key(candidate);
                    for (std::size_t j = 0; j < dim_; ++j) {
                        dm += conjugate_multiply(key[j], fp[j]);
                    }
                    const long double padic_match = 0.5L + 0.5L * cosine(current_padic, candidate_padic(candidate));
                    const long double observations =
                        static_cast<long double>(std::max<std::size_t>(1, oscs_[i].observations));
                    const long double frequency_pressure =
                        0.03L + (dimension_interference_ > 0.0L ? 0.75L * dimension_interference_ : 0.0L);
                    const long double frequency_penalty = 1.0L / std::sqrt(1.0L + frequency_pressure * observations);
                    const long double content_gain =
                        std::clamp(static_cast<long double>(oscs_[i].token.size()) / 7.0L, 0.55L, 1.35L);
                    const long double lexical_match = tail_overlap(active_tail, candidate_tail(candidate));
                    const long double prompt_fit =
                        dimension_interference_ > 0.0L ? normalized_complex_similarity(prompt_trace, key) : 0.0L;
                    const long double prompt_padic_fit =
                        dimension_interference_ > 0.0L
                            ? std::max<long double>(0.0L,
                                                    normalized_cosine(prompt_padic_trace, candidate_padic(candidate)))
                            : 0.0L;
                    const long double delta_fit =
                        dimension_interference_ > 0.0L && !prompt_delta.empty()
                            ? normalized_complex_similarity(prompt_delta, key)
                            : 0.0L;
                    const long double delta_padic_fit =
                        dimension_interference_ > 0.0L && !prompt_padic_delta.empty()
                            ? std::max<long double>(0.0L,
                                                    normalized_cosine(prompt_padic_delta, candidate_padic(candidate)))
                            : 0.0L;
                    const long double attractor_fit =
                        dimension_interference_ > 0.0L && !attractor_center.empty()
                            ? normalized_complex_similarity(attractor_center, key)
                            : 0.0L;
                    const long double attractor_padic_fit =
                        dimension_interference_ > 0.0L && !attractor_padic_center.empty()
                            ? 0.5L + 0.5L * normalized_cosine(attractor_padic_center, candidate_padic(candidate))
                            : 0.0L;
                    const long double attractor_pressure =
                        std::clamp(0.72L * attractor_fit + 0.28L * attractor_padic_fit, 0.0L, 1.0L);
                    long double axis_best = 0.0L;
                    long double axis_mean = 0.0L;
                    if (dimension_interference_ > 0.0L && !prompt_axes.empty()) {
                        for (const auto& axis : prompt_axes) {
                            const long double axis_field = normalized_complex_similarity(axis.field, key);
                            const long double axis_padic =
                                std::max<long double>(0.0L,
                                                      normalized_cosine(axis.padic, candidate_padic(candidate)));
                            const long double axis_score = axis.weight * (0.88L * axis_field + 0.12L * axis_padic);
                            axis_best = std::max(axis_best, axis_score);
                            axis_mean += axis_score;
                        }
                        axis_mean /= static_cast<long double>(prompt_axes.size());
                    }
                    const long double axis_drive =
                        dimension_interference_ > 0.0L
                            ? std::max<long double>(0.0L,
                                                    axis_best - 0.72L * axis_mean - 0.42L * attractor_pressure)
                            : 0.0L;
                    const long double prompt_specificity =
                        std::max<long double>(0.0L,
                                              0.40L * prompt_fit + 0.16L * prompt_padic_fit +
                                                  0.28L * delta_fit + 0.08L * delta_padic_fit +
                                                  0.34L * axis_drive -
                                                  0.62L * attractor_fit - 0.18L * attractor_padic_fit);
                    const long double differential_drive =
                        dimension_interference_ > 0.0L
                            ? std::max<long double>(0.0L,
                                                    std::max(0.68L * delta_fit + 0.32L * delta_padic_fit,
                                                             axis_drive) -
                                                        0.35L * attractor_pressure)
                            : 0.0L;
                    const long double field_drive =
                        dimension_interference_ > 0.0L
                            ? std::max<long double>(std::abs(dm),
                                                    (0.16L + 1.85L * dimension_interference_) *
                                                        differential_drive)
                            : std::abs(dm);
                    const long double context_gate =
                        dimension_interference_ > 0.0L
                            ? std::clamp(0.10L + 1.50L * lexical_match + 1.15L * prompt_specificity,
                                         0.05L,
                                         2.10L)
                            : 0.40L + 0.90L * lexical_match;
                    const long double bridge_fit =
                        projected_bridge_similarity(fp, candidate_transition(candidate), candidate_query(candidate));
                    const long double negative_spectral = complex_similarity(fp, candidate_negative_key(candidate));
                    const long double negative_padic =
                        0.5L + 0.5L * cosine(current_padic, candidate_negative_padic(candidate));
                    const long double collision =
                        std::clamp(0.78L * negative_spectral + 0.22L * negative_padic, 0.0L, 1.0L);
                    const long double contrastive_penalty =
                        std::clamp(1.0L - contrastive_strength_ * collision, 0.12L, 1.0L);
                    const long double reliability =
                        oscs_[i].strength * frequency_penalty * content_gain / (1.0L + oscs_[i].error_ema);
                    const long double route_gain =
                        dimension_interference_ > 0.0L
                            ? std::clamp(0.25L + 26.0L * dimension_interference_ * prompt_specificity,
                                         0.08L,
                                         4.60L)
                            : 1.0L;
                    const long double anti_template =
                        dimension_interference_ > 0.0L
                            ? std::clamp(1.18L - (0.85L + 1.80L * dimension_interference_) * attractor_pressure,
                                         0.05L,
                                         1.18L)
                            : 1.0L;
                    long double score =
                        field_drive * (0.70L + 0.30L * padic_match) *
                        (0.75L + 0.25L * bridge_fit) * reliability * context_gate *
                        contrastive_penalty * route_gain * anti_template;
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
                        cross += conjugate_multiply(q[j], qu[j]);
                        n1 += complex_norm(q[j]);
                        n2 += complex_norm(qu[j]);
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
            if (!is_subword_continuation(oscs_[best_i].token)) {
                const auto surface = subword_surface(oscs_[best_i].token);
                if (!out.empty()) out += ' ';
                out += surface;
            }
            used.insert(oscs_[best_i].token);
            push_active_token(oscs_[best_i].token);
            const auto bridged = apply_bridge(fp, candidate_transition(best));
            const auto& query = candidate_query(best);
            const PromptAxis* active_axis = nullptr;
            if (dimension_interference_ > 0.0L && !prompt_axes.empty()) {
                active_axis = &prompt_axes[(s + stable_hash(oscs_[best_i].token)) % prompt_axes.size()];
            }
            for (std::size_t j = 0; j < dim_; ++j) {
                const long double trace = dimension_interference_ > 0.0L
                                              ? 0.16L / (1.0L + 0.12L * static_cast<long double>(s))
                                              : 0.22L / (1.0L + 0.18L * static_cast<long double>(s));
                const long double delta_trace =
                    dimension_interference_ > 0.0L && !prompt_delta.empty()
                        ? std::min<long double>(0.34L, 0.08L + 0.82L * dimension_interference_) /
                              (1.0L + 0.04L * static_cast<long double>(s))
                        : 0.0L;
                const long double axis_trace =
                    active_axis != nullptr
                        ? std::min<long double>(0.32L, 0.06L + 0.70L * dimension_interference_) /
                              (1.0L + 0.03L * static_cast<long double>(s))
                        : 0.0L;
                const long double query_weight = dimension_interference_ > 0.0L ? 0.48L : 0.58L;
                const long double bridge_weight =
                    std::max(0.0L, 1.0L - query_weight - trace - delta_trace - axis_trace);
                fp[j] = query_weight * query[j] + bridge_weight * bridged[j] + trace * prompt_trace[j] +
                        delta_trace * prompt_delta[j] +
                        (active_axis != nullptr ? axis_trace * active_axis->field[j] : cx(0, 0));
            }
            const auto& next_padic = candidate_next_padic(best);
            for (std::size_t j = 0; j < std::min(current_padic.size(), next_padic.size()); ++j) {
                const long double trace = 0.18L / (1.0L + 0.20L * static_cast<long double>(s));
                const long double delta_trace =
                    dimension_interference_ > 0.0L && !prompt_padic_delta.empty()
                        ? std::min<long double>(0.22L, 0.04L + 0.46L * dimension_interference_) /
                              (1.0L + 0.04L * static_cast<long double>(s))
                        : 0.0L;
                const long double axis_trace =
                    active_axis != nullptr
                        ? std::min<long double>(0.20L, 0.04L + 0.40L * dimension_interference_) /
                              (1.0L + 0.03L * static_cast<long double>(s))
                        : 0.0L;
                const long double next_weight = std::max(0.0L, 1.0L - trace - delta_trace - axis_trace);
                current_padic[j] = next_weight * next_padic[j] + trace * prompt_padic_trace[j] +
                                   delta_trace * prompt_padic_delta[j] +
                                   (active_axis != nullptr ? axis_trace * active_axis->padic[j] : 0.0L);
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
    std::size_t dimensions() const noexcept { return dim_; }
    std::size_t oscillator_limit() const noexcept { return max_osc_; }

    std::vector<TokenSummary> token_summaries(std::size_t limit = 0) const {
        std::vector<TokenSummary> summaries;
        summaries.reserve(oscs_.size());
        for (const auto& oscillator : oscs_) {
            summaries.push_back({oscillator.token,
                                 oscillator.observations,
                                 oscillator.prototypes.size(),
                                 oscillator.strength,
                                 oscillator.error_ema});
        }
        std::sort(summaries.begin(), summaries.end(), [](const auto& left, const auto& right) {
            const long double left_rank = left.strength * std::log1p(static_cast<long double>(left.observations));
            const long double right_rank = right.strength * std::log1p(static_cast<long double>(right.observations));
            if (left_rank == right_rank) {
                return left.token < right.token;
            }
            return left_rank > right_rank;
        });
        if (limit != 0 && summaries.size() > limit) {
            summaries.resize(limit);
        }
        return summaries;
    }

    std::vector<TokenLink> nearest_token_links(std::string_view token, std::size_t limit = 16) const {
        const auto found = token_index_.find(std::string(token));
        if (found == token_index_.end()) {
            return {};
        }
        const auto& anchor = oscs_[found->second];
        std::vector<TokenLink> links;
        links.reserve(oscs_.size());
        for (std::size_t i = 0; i < oscs_.size(); ++i) {
            if (i == found->second) {
                continue;
            }
            const auto& candidate = oscs_[i];
            const long double next_similarity = complex_similarity(anchor.query, candidate.key);
            const long double context_similarity = complex_similarity(anchor.key, candidate.key);
            const long double transition_similarity = complex_similarity(anchor.transition, candidate.transition);
            const long double padic_similarity =
                0.5L + 0.5L * cosine(anchor.query_padic_signature, candidate.padic_signature);
            const long double reliability = 0.65L + 0.35L * std::clamp(candidate.strength / 4.0L, 0.0L, 1.0L);
            const long double association_score =
                reliability *
                (0.45L * next_similarity +
                 0.25L * context_similarity +
                 0.20L * transition_similarity +
                 0.10L * padic_similarity);
            if (association_score > 1.0e-12L) {
                links.push_back({candidate.token,
                                 candidate.observations,
                                 association_score,
                                 next_similarity,
                                 context_similarity,
                                 transition_similarity,
                                 padic_similarity});
            }
        }
        std::sort(links.begin(), links.end(), [](const auto& left, const auto& right) {
            if (left.association_score == right.association_score) {
                return left.token < right.token;
            }
            return left.association_score > right.association_score;
        });
        if (limit != 0 && links.size() > limit) {
            links.resize(limit);
        }
        return links;
    }

    void set_generation_temperature(long double temperature) noexcept {
        generation_temperature_ = std::clamp(temperature, 0.0L, 2.0L);
    }

    void set_learning_rate(long double rate) noexcept {
        learning_rate_ = std::clamp(rate, 1.0e-4L, 1.0L);
    }

    void set_update_probability(long double probability) noexcept {
        update_probability_ = std::clamp(probability, 0.0L, 1.0L);
    }

    void set_update_noise(long double scale) noexcept {
        update_noise_ = std::clamp(scale, 0.0L, 0.25L);
    }

    void set_random_init_scale(long double scale) noexcept {
        random_init_scale_ = std::clamp(scale, 0.0L, 0.25L);
    }

    void set_dimension_interference(long double strength) noexcept {
        dimension_interference_ = std::clamp(strength, 0.0L, 0.25L);
    }

    long double dimension_interference() const noexcept { return dimension_interference_; }

    void set_thread_count(std::size_t count) noexcept {
        const auto fallback = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        thread_count_ = count == 0 ? fallback : std::clamp<std::size_t>(count, 1, 1024);
        range_pool_.reset();
    }

    std::size_t thread_count() const noexcept { return thread_count_; }

    void set_parallel_min_dimensions(std::size_t dimensions) noexcept {
        parallel_min_dimensions_ = std::max<std::size_t>(1, dimensions);
    }

    void save_model(std::string_view path) const {
        std::ofstream output(std::string(path), std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot open model for write: " + std::string(path));
        }
        write_model(output);
        if (!output) {
            throw std::runtime_error("failed to write model: " + std::string(path));
        }
    }

    void load_model(std::string_view path) {
        std::ifstream input(std::string(path), std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open model for read: " + std::string(path));
        }
        read_model(input);
        if (!input.eof() && !input) {
            throw std::runtime_error("failed to read model: " + std::string(path));
        }
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

    static constexpr std::string_view model_magic() noexcept { return "DZETA_OSC_FIELD"; }
    static constexpr std::uint32_t model_version() noexcept { return 2U; }
    static constexpr std::size_t max_serialized_string_bytes = 64U * 1024U * 1024U;
    static constexpr std::size_t max_serialized_vector_items = 16U * 1024U * 1024U;

    template <typename T>
    static void write_pod(std::ostream& output, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T>
    static void read_pod(std::istream& input, T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        input.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    static void write_count(std::ostream& output, std::size_t value) {
        const auto stored = static_cast<std::uint64_t>(value);
        write_pod(output, stored);
    }

    static std::size_t read_count(std::istream& input, std::string_view label) {
        std::uint64_t stored = 0;
        read_pod(input, stored);
        if (!input) {
            throw std::runtime_error("truncated model count: " + std::string(label));
        }
        if (stored > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("model count is too large: " + std::string(label));
        }
        return static_cast<std::size_t>(stored);
    }

    static void write_string(std::ostream& output, std::string_view value) {
        write_count(output, value.size());
        output.write(value.data(), static_cast<std::streamsize>(value.size()));
    }

    static void read_string(std::istream& input, std::string& value, std::string_view label) {
        const std::size_t size = read_count(input, label);
        if (size > max_serialized_string_bytes) {
            throw std::runtime_error("model string is too large: " + std::string(label));
        }
        value.resize(size);
        input.read(value.data(), static_cast<std::streamsize>(size));
        if (!input) {
            throw std::runtime_error("truncated model string: " + std::string(label));
        }
    }

    template <typename T>
    static void write_scalar_vector(std::ostream& output, const std::vector<T>& values) {
        write_count(output, values.size());
        for (const auto& value : values) {
            write_pod(output, value);
        }
    }

    template <typename T>
    static void read_scalar_vector(std::istream& input,
                                   std::vector<T>& values,
                                   std::string_view label) {
        const std::size_t size = read_count(input, label);
        if (size > max_serialized_vector_items) {
            throw std::runtime_error("model vector is too large: " + std::string(label));
        }
        values.resize(size);
        for (auto& value : values) {
            read_pod(input, value);
        }
        if (!input) {
            throw std::runtime_error("truncated model vector: " + std::string(label));
        }
    }

    static void write_complex_vector(std::ostream& output, const std::vector<cx>& values) {
        write_count(output, values.size());
        for (const auto& value : values) {
            const long double real = value.real();
            const long double imag = value.imag();
            write_pod(output, real);
            write_pod(output, imag);
        }
    }

    static void read_complex_vector(std::istream& input,
                                    std::vector<cx>& values,
                                    std::string_view label) {
        const std::size_t size = read_count(input, label);
        if (size > max_serialized_vector_items) {
            throw std::runtime_error("model complex vector is too large: " + std::string(label));
        }
        values.resize(size);
        for (auto& value : values) {
            long double real = 0.0L;
            long double imag = 0.0L;
            read_pod(input, real);
            read_pod(input, imag);
            value = cx(real, imag);
        }
        if (!input) {
            throw std::runtime_error("truncated model complex vector: " + std::string(label));
        }
    }

    static void require_dimension(std::size_t size, std::size_t expected, std::string_view label) {
        if (size != expected) {
            throw std::runtime_error("model dimension mismatch in " + std::string(label));
        }
    }

    static void write_context_prototype(std::ostream& output, const ContextPrototype& prototype) {
        write_complex_vector(output, prototype.key);
        write_complex_vector(output, prototype.query);
        write_complex_vector(output, prototype.transition);
        write_scalar_vector(output, prototype.padic_signature);
        write_scalar_vector(output, prototype.query_padic_signature);
        write_complex_vector(output, prototype.negative_key);
        write_scalar_vector(output, prototype.negative_padic_signature);
        write_scalar_vector(output, prototype.context_tail);
        write_pod(output, prototype.error_ema);
        write_count(output, prototype.observations);
    }

    ContextPrototype read_context_prototype(std::istream& input) const {
        ContextPrototype prototype;
        read_complex_vector(input, prototype.key, "prototype.key");
        read_complex_vector(input, prototype.query, "prototype.query");
        read_complex_vector(input, prototype.transition, "prototype.transition");
        read_scalar_vector(input, prototype.padic_signature, "prototype.padic_signature");
        read_scalar_vector(input, prototype.query_padic_signature, "prototype.query_padic_signature");
        read_complex_vector(input, prototype.negative_key, "prototype.negative_key");
        read_scalar_vector(input, prototype.negative_padic_signature, "prototype.negative_padic_signature");
        read_scalar_vector(input, prototype.context_tail, "prototype.context_tail");
        read_pod(input, prototype.error_ema);
        prototype.observations = read_count(input, "prototype.observations");
        require_dimension(prototype.key.size(), dim_, "prototype.key");
        require_dimension(prototype.query.size(), dim_, "prototype.query");
        require_dimension(prototype.transition.size(), dim_, "prototype.transition");
        require_dimension(prototype.padic_signature.size(), dim_, "prototype.padic_signature");
        require_dimension(prototype.query_padic_signature.size(), dim_, "prototype.query_padic_signature");
        require_dimension(prototype.negative_key.size(), dim_, "prototype.negative_key");
        require_dimension(prototype.negative_padic_signature.size(), dim_, "prototype.negative_padic_signature");
        return prototype;
    }

    static void write_token_oscillator(std::ostream& output, const TokenOscillator& oscillator) {
        write_string(output, oscillator.token);
        write_complex_vector(output, oscillator.query);
        write_complex_vector(output, oscillator.key);
        write_complex_vector(output, oscillator.transition);
        write_scalar_vector(output, oscillator.padic_signature);
        write_scalar_vector(output, oscillator.query_padic_signature);
        write_complex_vector(output, oscillator.negative_key);
        write_scalar_vector(output, oscillator.negative_padic_signature);
        write_pod(output, oscillator.strength);
        write_pod(output, oscillator.error_ema);
        write_count(output, oscillator.observations);
        write_count(output, oscillator.prototypes.size());
        for (const auto& prototype : oscillator.prototypes) {
            write_context_prototype(output, prototype);
        }
    }

    TokenOscillator read_token_oscillator(std::istream& input) const {
        TokenOscillator oscillator;
        read_string(input, oscillator.token, "oscillator.token");
        read_complex_vector(input, oscillator.query, "oscillator.query");
        read_complex_vector(input, oscillator.key, "oscillator.key");
        read_complex_vector(input, oscillator.transition, "oscillator.transition");
        read_scalar_vector(input, oscillator.padic_signature, "oscillator.padic_signature");
        read_scalar_vector(input, oscillator.query_padic_signature, "oscillator.query_padic_signature");
        read_complex_vector(input, oscillator.negative_key, "oscillator.negative_key");
        read_scalar_vector(input, oscillator.negative_padic_signature, "oscillator.negative_padic_signature");
        read_pod(input, oscillator.strength);
        read_pod(input, oscillator.error_ema);
        oscillator.observations = read_count(input, "oscillator.observations");
        const std::size_t prototype_count = read_count(input, "oscillator.prototypes");
        if (prototype_count > 1024U) {
            throw std::runtime_error("model has too many prototypes for token: " + oscillator.token);
        }
        oscillator.prototypes.reserve(prototype_count);
        for (std::size_t i = 0; i < prototype_count; ++i) {
            oscillator.prototypes.push_back(read_context_prototype(input));
        }
        require_dimension(oscillator.query.size(), dim_, "oscillator.query");
        require_dimension(oscillator.key.size(), dim_, "oscillator.key");
        require_dimension(oscillator.transition.size(), dim_, "oscillator.transition");
        require_dimension(oscillator.padic_signature.size(), dim_, "oscillator.padic_signature");
        require_dimension(oscillator.query_padic_signature.size(), dim_, "oscillator.query_padic_signature");
        require_dimension(oscillator.negative_key.size(), dim_, "oscillator.negative_key");
        require_dimension(oscillator.negative_padic_signature.size(), dim_, "oscillator.negative_padic_signature");
        return oscillator;
    }

    void write_model(std::ostream& output) const {
        write_string(output, model_magic());
        write_pod(output, model_version());
        write_count(output, max_osc_);
        write_count(output, dim_);
        write_count(output, thread_count_);
        write_count(output, parallel_min_dimensions_);
        write_pod(output, learning_rate_);
        write_pod(output, generation_temperature_);
        write_pod(output, contrastive_rate_);
        write_pod(output, contrastive_margin_);
        write_pod(output, contrastive_strength_);
        write_pod(output, update_probability_);
        write_pod(output, update_noise_);
        write_pod(output, random_init_scale_);
        write_pod(output, dimension_interference_);
        write_count(output, max_prototypes_per_token_);
        write_count(output, max_hard_negatives_);
        write_count(output, max_context_tokens_);
        write_count(output, contrastive_period_);
        write_count(output, total_observations_);
        write_count(output, contrastive_updates_);
        write_count(output, loss_updates_);
        write_pod(output, loss_ema_);

        std::ostringstream rng_state;
        rng_state << rng_;
        write_string(output, rng_state.str());

        write_count(output, oscs_.size());
        for (const auto& oscillator : oscs_) {
            write_token_oscillator(output, oscillator);
        }
    }

    void read_model(std::istream& input) {
        std::string magic;
        read_string(input, magic, "model.magic");
        std::uint32_t version = 0;
        read_pod(input, version);
        if (magic != model_magic()) {
            throw std::runtime_error("not a dzeta oscillator model");
        }
        if (version == 0 || version > model_version()) {
            throw std::runtime_error("unsupported dzeta oscillator model version");
        }

        max_osc_ = std::max<std::size_t>(128, read_count(input, "max_osc"));
        dim_ = std::max<std::size_t>(16, read_count(input, "dimensions"));
        thread_count_ = std::max<std::size_t>(1, read_count(input, "thread_count"));
        range_pool_.reset();
        parallel_min_dimensions_ = std::max<std::size_t>(1, read_count(input, "parallel_min_dimensions"));
        read_pod(input, learning_rate_);
        read_pod(input, generation_temperature_);
        read_pod(input, contrastive_rate_);
        read_pod(input, contrastive_margin_);
        read_pod(input, contrastive_strength_);
        read_pod(input, update_probability_);
        read_pod(input, update_noise_);
        read_pod(input, random_init_scale_);
        if (version >= 2U) {
            read_pod(input, dimension_interference_);
        } else {
            dimension_interference_ = 0.0L;
        }
        max_prototypes_per_token_ = std::max<std::size_t>(1, read_count(input, "max_prototypes_per_token"));
        max_hard_negatives_ = std::max<std::size_t>(1, read_count(input, "max_hard_negatives"));
        max_context_tokens_ = std::max<std::size_t>(1, read_count(input, "max_context_tokens"));
        contrastive_period_ = std::max<std::size_t>(1, read_count(input, "contrastive_period"));
        total_observations_ = read_count(input, "total_observations");
        contrastive_updates_ = read_count(input, "contrastive_updates");
        loss_updates_ = read_count(input, "loss_updates");
        read_pod(input, loss_ema_);

        std::string rng_state;
        read_string(input, rng_state, "rng_state");
        std::istringstream rng_input(rng_state);
        rng_input >> rng_;
        if (!rng_input) {
            rng_.seed(entropy_seed());
        }

        initialize_spectral_basis();
        initialize_seed_projection(64);
        fp_cx_.clear();

        const std::size_t oscillator_count = read_count(input, "oscillators");
        if (oscillator_count > max_osc_) {
            max_osc_ = oscillator_count;
        }
        oscs_.clear();
        token_index_.clear();
        oscs_.reserve(oscillator_count);
        for (std::size_t i = 0; i < oscillator_count; ++i) {
            auto oscillator = read_token_oscillator(input);
            if (oscillator.token.empty()) {
                throw std::runtime_error("model contains an empty token");
            }
            if (token_index_.find(oscillator.token) != token_index_.end()) {
                throw std::runtime_error("model contains a duplicate token: " + oscillator.token);
            }
            token_index_[oscillator.token] = oscs_.size();
            oscs_.push_back(std::move(oscillator));
        }
    }

    TokenOscillator make_token_oscillator(std::string token) const {
        TokenOscillator oscillator{std::move(token),
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
        if (random_init_scale_ > 0.0L) {
            add_complex_noise(oscillator.key, random_init_scale_);
            add_complex_noise(oscillator.query, random_init_scale_);
            add_complex_noise(oscillator.transition, random_init_scale_);
        }
        return oscillator;
    }

    void initialize_spectral_basis() {
        steps_.resize(dim_);
        const std::size_t nz = zeta_zero_count();
        zeta_basis_.resize(dim_);
        for (std::size_t z = 0; z < dim_; ++z) {
            if (z < 64) steps_[z] = 1;
            else if (z < 128) steps_[z] = 4;
            else steps_[z] = std::max<std::size_t>(1, nz / dim_);
            zeta_basis_[z] = zeta_zero((z * steps_[z]) % nz);
        }
    }

    void initialize_seed_projection(std::size_t count) {
        seed_primes_ = generate_first_primes(std::max<std::size_t>(1, count));
        seed_theta_.resize(seed_primes_.size());
        seed_energy_.resize(seed_primes_.size());
        seed_padic_log_.resize(seed_primes_.size());
        seed_prime_phase_.resize(seed_primes_.size());
        for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
            seed_theta_[i] = riemann_siegel_theta(seed_primes_[i]);
            seed_energy_[i] = spectral_energy(seed_primes_[i], 32);
            seed_padic_log_[i] = std::log1p(static_cast<long double>(seed_primes_[i] % 997U));
            seed_prime_phase_[i] = static_cast<long double>(seed_primes_[i]) * 0.0174533L;
        }
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

    static bool hardware_random64(std::uint64_t& value) noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
        return rdrand64_gcc(value);
#elif defined(_MSC_VER) && defined(_M_X64)
        unsigned __int64 generated = 0;
        for (int attempt = 0; attempt < 8; ++attempt) {
            if (_rdrand64_step(&generated) != 0) {
                value = static_cast<std::uint64_t>(generated);
                return true;
            }
        }
        return false;
#else
        (void)value;
        return false;
#endif
    }

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    __attribute__((target("rdrnd"))) static bool rdrand64_gcc(std::uint64_t& value) noexcept {
        unsigned long long generated = 0;
        for (int attempt = 0; attempt < 8; ++attempt) {
            if (_rdrand64_step(&generated) != 0) {
                value = static_cast<std::uint64_t>(generated);
                return true;
            }
        }
        return false;
    }
#endif

    static std::uint64_t entropy_seed() {
        std::random_device rd;
        const auto now = static_cast<std::uint64_t>(std::time(nullptr));
        const auto tick =
            static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::uint64_t hardware = 0;
        (void)hardware_random64(hardware);
        return hardware ^
               (static_cast<std::uint64_t>(rd()) << 32U) ^
               static_cast<std::uint64_t>(rd()) ^
               (now * 0x9e3779b97f4a7c15ULL) ^
               (tick * 0xbf58476d1ce4e5b9ULL);
    }

    static std::size_t env_size_or(const char* name, std::size_t fallback) noexcept {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') {
            return fallback;
        }
        char* end = nullptr;
        const auto parsed = std::strtoull(value, &end, 10);
        if (end == value || parsed == 0ULL) {
            return fallback;
        }
        return static_cast<std::size_t>(std::min<unsigned long long>(parsed, 1024ULL));
    }

    static std::size_t default_thread_count() noexcept {
        const auto hardware = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        return env_size_or("DZETA_THREADS", hardware);
    }

    static std::size_t default_parallel_min_dimensions() noexcept {
        return env_size_or("DZETA_PARALLEL_MIN_DIM", 2048);
    }

    std::size_t effective_thread_count(std::size_t work_items) const noexcept {
        if (thread_count_ <= 1 || work_items < parallel_min_dimensions_ || work_items <= 1) {
            return 1;
        }
        return std::min(thread_count_, work_items);
    }

    RangeThreadPool& range_pool(std::size_t workers) const {
        if (!range_pool_ || range_pool_->max_workers() != workers) {
            range_pool_ = std::make_unique<RangeThreadPool>(workers);
        }
        return *range_pool_;
    }

    template <typename Fn>
    void parallel_for_ranges(std::size_t work_items, Fn&& fn) const {
        const std::size_t workers = effective_thread_count(work_items);
        if (workers <= 1) {
            fn(0, work_items);
            return;
        }
        range_pool(workers).run(work_items, workers, std::forward<Fn>(fn));
    }

    static long double complex_norm(cx value) noexcept {
        const long double re = value.real();
        const long double im = value.imag();
        return re * re + im * im;
    }

    static cx conjugate_multiply(cx left, cx right) noexcept {
        const long double lr = left.real();
        const long double li = left.imag();
        const long double rr = right.real();
        const long double ri = right.imag();
        return {lr * rr + li * ri, lr * ri - li * rr};
    }

    static void sincos_ld(long double value, long double& sin_value, long double& cos_value) noexcept {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_sincosl(value, &sin_value, &cos_value);
#else
        sin_value = std::sin(value);
        cos_value = std::cos(value);
#endif
    }

    void seed_weyl_transform_into(std::string_view impulse,
                                  std::vector<cx>& ampl,
                                  std::vector<long double>& padic,
                                  bool enable_dimensional_interference = true,
                                  bool allow_parallel = true) const {
        ampl.resize(dim_);
        padic.assign(dim_, 0.0L);
        if (seed_primes_.empty()) {
            return;
        }

        const auto signature = field_impulse_signature(impulse, seed_primes_.size());
        std::vector<long double> seed_weight(signature.size(), 0.0L);
        long double padic_base = 0.0L;
        for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
            const long double charge = signature[i];
            const long double act = std::clamp(0.45L + 0.35L * std::abs(charge), 0.0L, 1.0L);
            const long double en = seed_energy_[i];
            seed_weight[i] = act * en;
            const long double padic_coordinate = seed_padic_log_[i] * (1.0L + charge);
            padic_base += act * en * padic_norm(padic_coordinate, seed_primes_[i % seed_primes_.size()] % 997U + 2U);
        }
        if (std::abs(padic_base) > 1.0e-30L) {
            std::fill(padic.begin(), padic.end(), padic_base);
        }

        const auto compute_range = [&](std::size_t begin, std::size_t end) {
            constexpr std::size_t seed_stack_capacity = 128;
            std::array<long double, seed_stack_capacity> dither_sin{};
            std::array<long double, seed_stack_capacity> dither_cos{};
            std::array<long double, seed_stack_capacity> step_sin{};
            std::array<long double, seed_stack_capacity> step_cos{};
            for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
                const long double charge = signature[i];
                const long double start_phase =
                    charge * (static_cast<long double>(begin) + 1.0L) * 12.9898L + seed_prime_phase_[i];
                const long double step_phase = charge * 12.9898L;
                sincos_ld(start_phase, dither_sin[i], dither_cos[i]);
                sincos_ld(step_phase, step_sin[i], step_cos[i]);
            }
            for (std::size_t z = begin; z < end; ++z) {
                long double sum_re = 0.0L;
                long double sum_im = 0.0L;
                const long double zr = zeta_basis_[z];
                for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
                    const long double charge = signature[i];
                    const long double theta = seed_theta_[i];
                    const long double phase_dither = 0.03L * dither_sin[i];
                    const long double phase = theta * zr + charge * 0.5L + phase_dither;
                    const long double weight = seed_weight[i];
                    long double sin_phase = 0.0L;
                    long double cos_phase = 1.0L;
                    sincos_ld(phase, sin_phase, cos_phase);
                    sum_re += weight * cos_phase;
                    sum_im += weight * sin_phase;

                    const long double next_sin =
                        dither_sin[i] * step_cos[i] + dither_cos[i] * step_sin[i];
                    const long double next_cos =
                        dither_cos[i] * step_cos[i] - dither_sin[i] * step_sin[i];
                    dither_sin[i] = next_sin;
                    dither_cos[i] = next_cos;
                }
                ampl[z] = cx(sum_re, sum_im);
            }
        };
        if (allow_parallel) {
            parallel_for_ranges(dim_, compute_range);
        } else {
            compute_range(0, dim_);
        }

        long double an = 0.0L;
        for (auto value : ampl) an += complex_norm(value);
        if (an > 1.0e-30L) {
            an = std::sqrt(an);
            for (auto& value : ampl) value /= an;
        }
        if (enable_dimensional_interference) {
            apply_dimensional_interference(impulse, signature, ampl);
        }
        long double pn = 0.0L;
        for (auto value : padic) pn += value * value;
        if (pn > 1.0e-30L) {
            pn = std::sqrt(pn);
            for (auto& value : padic) value /= pn;
        }
    }

    std::pair<std::vector<cx>, std::vector<long double>> seed_weyl_transform(std::string_view impulse,
                                                                             bool allow_parallel = true) const {
        std::vector<cx> ampl;
        std::vector<long double> padic;
        seed_weyl_transform_into(impulse, ampl, padic, true, allow_parallel);
        return {std::move(ampl), std::move(padic)};
    }

    void apply_dimensional_interference(std::string_view impulse,
                                        const std::vector<long double>& signature,
                                        std::vector<cx>& ampl) const {
        if (dimension_interference_ <= 0.0L || ampl.size() < 8 || signature.empty()) {
            return;
        }

        const std::vector<cx> source = ampl;
        const std::size_t count = source.size();
        std::uint64_t state =
            stable_hash(impulse) ^ (static_cast<std::uint64_t>(count) * 0x9e3779b97f4a7c15ULL);
        auto next_shift = [&]() {
            return static_cast<std::size_t>(1U + (splitmix64(state) % (count - 1U)));
        };
        const std::size_t shift_a = next_shift();
        const std::size_t shift_b = next_shift();
        const std::size_t shift_c = next_shift();
        std::size_t stride = next_shift();
        if ((stride & 1U) == 0U) {
            ++stride;
        }
        if (stride >= count) {
            stride = 1U;
        }

        const long double dim_gain =
            std::sqrt(std::log2(static_cast<long double>(count) + 2.0L) / std::log2(194.0L));
        const long double strength = std::min<long double>(0.65L, dimension_interference_ * dim_gain);
        const long double fold_scale = std::sqrt(static_cast<long double>(count));

        for (std::size_t z = 0; z < count; ++z) {
            const std::size_t ia = (z + shift_a) % count;
            const std::size_t ib = (z * stride + shift_b) % count;
            const std::size_t ic = (z + shift_c) % count;
            const long double charge = signature[z % signature.size()];
            const long double gate = 0.55L + 0.45L * std::abs(charge);
            const cx phase_gate(1.0L + 0.15L * charge, 0.35L * charge);
            const cx folded =
                fold_scale * (source[ia] * std::conj(source[ib]) + 0.5L * source[z] * std::conj(source[ic]));
            ampl[z] = source[z] + strength * gate * phase_gate * folded;
        }
        normalize_complex(ampl);
    }

    const TokenOscillator* find_prompt_oscillator(std::string token) const {
        auto found = token_index_.find(token);
        if (found != token_index_.end()) {
            return &oscs_[found->second];
        }
        if (!token.empty()) {
            token[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(token[0])));
            found = token_index_.find(token);
            if (found != token_index_.end()) {
                return &oscs_[found->second];
            }
        }
        return nullptr;
    }

    void build_attractor_center(std::vector<cx>& center, std::vector<long double>& padic_center) const {
        center.assign(dim_, cx(0, 0));
        padic_center.assign(dim_, 0.0L);
        long double total_weight = 0.0L;
        for (const auto& oscillator : oscs_) {
            if (oscillator.observations == 0 || oscillator.token.size() <= 1 ||
                is_subword_continuation(oscillator.token)) {
                continue;
            }
            const long double observations = static_cast<long double>(oscillator.observations);
            const long double weight =
                std::sqrt(observations) * std::clamp(oscillator.strength / 4.0L, 0.10L, 2.00L);
            if (weight <= 1.0e-18L) {
                continue;
            }
            for (std::size_t j = 0; j < dim_; ++j) {
                center[j] += weight * (0.55L * oscillator.key[j] + 0.45L * oscillator.query[j]);
                padic_center[j] +=
                    weight * (0.55L * oscillator.padic_signature[j] + 0.45L * oscillator.query_padic_signature[j]);
            }
            total_weight += weight;
        }
        if (total_weight <= 1.0e-18L) {
            center.clear();
            padic_center.clear();
            return;
        }
        const long double inv_weight = 1.0L / total_weight;
        for (auto& value : center) {
            value *= inv_weight;
        }
        for (auto& value : padic_center) {
            value *= inv_weight;
        }
        normalize_complex(center);
        normalize_real(padic_center);
    }

    static void remove_attractor_projection(std::vector<cx>& values, const std::vector<cx>& center) {
        const std::size_t count = std::min(values.size(), center.size());
        if (count == 0) {
            return;
        }
        cx projection = 0;
        for (std::size_t i = 0; i < count; ++i) {
            projection += conjugate_multiply(center[i], values[i]);
        }
        for (std::size_t i = 0; i < count; ++i) {
            values[i] -= projection * center[i];
        }
        normalize_complex(values);
    }

    static void remove_attractor_projection(std::vector<long double>& values,
                                            const std::vector<long double>& center) {
        const std::size_t count = std::min(values.size(), center.size());
        if (count == 0) {
            return;
        }
        long double projection = 0.0L;
        for (std::size_t i = 0; i < count; ++i) {
            projection += values[i] * center[i];
        }
        for (std::size_t i = 0; i < count; ++i) {
            values[i] -= projection * center[i];
        }
        normalize_real(values);
    }

    void inject_prompt_resonance(std::string_view text,
                                 std::vector<cx>& state,
                                 std::vector<long double>& padic) const {
        if (dimension_interference_ <= 0.0L || state.size() != dim_) {
            return;
        }
        const auto tokens = tokenize_query(text);
        if (tokens.empty()) {
            return;
        }
        std::vector<const TokenOscillator*> anchors;
        anchors.reserve(tokens.size());
        for (const auto& token : tokens) {
            if (token.size() <= 1) {
                continue;
            }
            if (const auto* oscillator = find_prompt_oscillator(token)) {
                anchors.push_back(oscillator);
            }
        }
        if (anchors.empty()) {
            return;
        }

        const long double base_weight =
            std::min<long double>(0.58L, dimension_interference_ * 1.85L) /
            std::sqrt(static_cast<long double>(anchors.size()));
        for (std::size_t a = 0; a < anchors.size(); ++a) {
            const auto& oscillator = *anchors[a];
            const long double recency =
                1.0L + static_cast<long double>(a) / static_cast<long double>(anchors.size());
            const long double weight = std::min<long double>(0.62L, base_weight * recency);
            for (std::size_t j = 0; j < dim_; ++j) {
                const cx anchor = 0.35L * oscillator.key[j] + 0.65L * oscillator.query[j];
                state[j] = (1.0L - weight) * state[j] + weight * anchor;
                padic[j] = (1.0L - weight) * padic[j] + weight * oscillator.query_padic_signature[j];
            }
            normalize_complex(state);
            normalize_real(padic);
        }
    }

    static void normalize_complex(std::vector<cx>& values) {
        long double norm = 0.0L;
        for (auto value : values) {
            norm += complex_norm(value);
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
            loss += complex_norm(left[i] - right[i]);
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
            dot += conjugate_multiply(left[i], right[i]);
            left_norm += complex_norm(left[i]);
            right_norm += complex_norm(right[i]);
        }
        if (left_norm <= 1.0e-30L || right_norm <= 1.0e-30L) {
            return 0.0L;
        }
        return std::clamp(std::abs(dot) / std::sqrt(left_norm * right_norm), 0.0L, 1.0L);
    }

    static long double normalized_complex_similarity(const std::vector<cx>& left, const std::vector<cx>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0L;
        }
        cx dot = 0;
        for (std::size_t i = 0; i < count; ++i) {
            dot += conjugate_multiply(left[i], right[i]);
        }
        return std::clamp(std::abs(dot), 0.0L, 1.0L);
    }

    static void spectral_bridge_into(const std::vector<cx>& from,
                                     const std::vector<cx>& to,
                                     std::vector<cx>& bridge) {
        const std::size_t count = std::min(from.size(), to.size());
        bridge.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            const long double magnitude = std::max<long double>(1.0e-12L, std::abs(from[i]));
            bridge[i] = to[i] * std::conj(from[i]) / magnitude;
        }
        normalize_complex(bridge);
    }

    static std::vector<cx> spectral_bridge(const std::vector<cx>& from, const std::vector<cx>& to) {
        std::vector<cx> bridge;
        spectral_bridge_into(from, to, bridge);
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

    static long double projected_bridge_similarity(const std::vector<cx>& state,
                                                   const std::vector<cx>& bridge,
                                                   const std::vector<cx>& target) {
        const std::size_t count = std::min({state.size(), bridge.size(), target.size()});
        if (count == 0) {
            return 0.0L;
        }
        cx dot = 0;
        long double projected_norm = 0.0L;
        long double target_norm = 0.0L;
        for (std::size_t i = 0; i < count; ++i) {
            const cx projected = state[i] * bridge[i];
            dot += conjugate_multiply(projected, target[i]);
            projected_norm += complex_norm(projected);
            target_norm += complex_norm(target[i]);
        }
        if (projected_norm <= 1.0e-30L || target_norm <= 1.0e-30L) {
            return 0.0L;
        }
        return std::clamp(std::abs(dot) / std::sqrt(projected_norm * target_norm), 0.0L, 1.0L);
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
            projection += conjugate_multiply(away[i], vector[i]);
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

    static long double normalized_cosine(const std::vector<long double>& left,
                                         const std::vector<long double>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0L;
        }
        long double dot = 0.0L;
        for (std::size_t i = 0; i < count; ++i) {
            dot += left[i] * right[i];
        }
        return std::clamp(dot, -1.0L, 1.0L);
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

    void add_complex_noise(std::vector<cx>& values, long double scale) const {
        if (scale <= 0.0L) {
            return;
        }
        for (auto& value : values) {
            value += cx(scale * centered_noise(), scale * centered_noise());
        }
        normalize_complex(values);
    }

    void add_real_noise(std::vector<long double>& values, long double scale) const {
        if (scale <= 0.0L) {
            return;
        }
        for (auto& value : values) {
            value += scale * centered_noise();
        }
        normalize_real(values);
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
            const long double spectral_match = normalized_complex_similarity(prototype.key, target_key);
            const long double padic_match = 0.5L + 0.5L * normalized_cosine(prototype.padic_signature, target_padic);
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

        const auto scan_range = [&](std::size_t begin, std::size_t end, std::vector<HardNegative>& out) {
            for (std::size_t i = begin; i < end; ++i) {
                if (i == positive_index || oscs_[i].observations == 0) {
                    continue;
                }
                const bool has_prototype = !oscs_[i].prototypes.empty();
                const std::size_t prototypes = std::max<std::size_t>(1, oscs_[i].prototypes.size());
                for (std::size_t p = 0; p < prototypes; ++p) {
                    const auto& key = has_prototype ? oscs_[i].prototypes[p].key : oscs_[i].key;
                    const auto& padic =
                        has_prototype ? oscs_[i].prototypes[p].padic_signature : oscs_[i].padic_signature;
                    const long double spectral_match = normalized_complex_similarity(key, target_key);
                    const long double padic_match = 0.5L + 0.5L * normalized_cosine(padic, target_padic);
                    const long double score = 0.84L * spectral_match + 0.16L * padic_match;
                    if (score > contrastive_margin_) {
                        out.push_back({score, i, has_prototype ? p : no_prototype});
                    }
                }
            }
        };

        const std::size_t scan_workers =
            (thread_count_ > 1 && dim_ >= parallel_min_dimensions_ && oscs_.size() >= thread_count_ * 2U)
                ? std::min(thread_count_, oscs_.size())
                : 1U;
        if (scan_workers <= 1) {
            scan_range(0, oscs_.size(), candidates);
        } else {
            std::mutex candidates_mutex;
            range_pool(scan_workers).run(oscs_.size(), scan_workers, [&](std::size_t begin, std::size_t end) {
                std::vector<HardNegative> local_candidates;
                local_candidates.reserve(std::min<std::size_t>(end - begin, 64));
                scan_range(begin, end, local_candidates);
                if (!local_candidates.empty()) {
                    std::lock_guard<std::mutex> lock(candidates_mutex);
                    candidates.insert(candidates.end(), local_candidates.begin(), local_candidates.end());
                }
            });
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
        const std::size_t field_items = std::min<std::size_t>(f.size(), 256);
        long double padic_base = 0.0L;
        for (std::size_t i = 0; i < field_items; ++i) {
            const long double act = f.activations[i];
            const long double en = f.energy[i];
            padic_base += act * en * padic_norm(f.padic_coordinates[i], f.primes[i % f.size()] % 997U + 2U);
        }
        if (std::abs(padic_base) > 1.0e-30L) {
            std::fill(padic.begin(), padic.end(), padic_base);
        }

        parallel_for_ranges(dim_, [&](std::size_t begin, std::size_t end) {
            for (std::size_t z = begin; z < end; ++z) {
                long double sum_re = 0.0L;
                long double sum_im = 0.0L;
                const long double zr = zeta_basis_[z];
                for (std::size_t i = 0; i < field_items; ++i) {
                    const long double act = f.activations[i];
                    const long double en = f.energy[i];
                    const long double theta = f.theta[i];
                    const long double charge = f.semantic_charge[i];
                    const long double phase_dither =
                        0.03L * std::sin(charge * (static_cast<long double>(z) + 1.0L) * 12.9898L +
                                          static_cast<long double>(f.primes[i]) * 0.0174533L);
                    long double im = 0.0L;
                    long double re = 1.0L;
                    sincos_ld(theta * zr + charge * 0.5L + phase_dither, im, re);
                    const long double weight = act * en;
                    sum_re += weight * re;
                    sum_im += weight * im;
                }
                ampl[z] = cx(sum_re, sum_im);
            }
        });
        long double an = 0;
        for (auto v : ampl) an += complex_norm(v);
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
    std::size_t thread_count_;
    std::size_t parallel_min_dimensions_;
    std::vector<std::size_t> steps_;
    std::vector<long double> zeta_basis_;
    std::vector<std::uint32_t> seed_primes_;
    std::vector<long double> seed_theta_;
    std::vector<long double> seed_energy_;
    std::vector<long double> seed_padic_log_;
    std::vector<long double> seed_prime_phase_;
    mutable std::mt19937_64 rng_;
    mutable std::unique_ptr<RangeThreadPool> range_pool_;
    long double learning_rate_ = 0.32L;
    long double generation_temperature_ = 0.08L;
    long double contrastive_rate_ = 0.08L;
    long double contrastive_margin_ = 0.62L;
    long double contrastive_strength_ = 0.74L;
    long double update_probability_ = 1.0L;
    long double update_noise_ = 0.0L;
    long double random_init_scale_ = 0.0L;
    long double dimension_interference_ = 0.0L;
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
