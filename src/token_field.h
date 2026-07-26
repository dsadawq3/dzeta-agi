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
#include <cstring>
#include <ctime>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <istream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <random>
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

// Hot-path floating type. long double forces x87 on x86-64 (no SIMD, high
// latency, ~4-6x slower dot products); double keeps the same math
// SIMD-friendly. Override with -DDZETA_REAL="long double" to restore the
// old numerics. The on-disk model format stays long double either way, so
// files remain interchangeable between builds (double -> long double
// widening is exact and round-trips).
#ifndef DZETA_REAL
#define DZETA_REAL double
#endif
using dz_real = DZETA_REAL;
using cx = std::complex<dz_real>;

inline dz_real padic_norm(dz_real x, std::uint32_t p) {
    if (std::abs(x) < 1e-30) return 0.0;
    dz_real v = 0, ax = std::abs(x);
    while (ax > 1.0) { ax /= p; v += 1.0; }
    while (ax > 0.0 && ax < 1.0) { ax *= p; v -= 1.0; }
    return std::pow(static_cast<dz_real>(p), -v);
}

class OscillatorField {
public:
    struct TokenSummary {
        std::string token;
        std::size_t observations = 0;
        std::size_t prototypes = 0;
        dz_real strength = 0.0;
        dz_real error_ema = 0.0;
    };

    struct TokenLink {
        std::string token;
        std::size_t observations = 0;
        dz_real association_score = 0.0;
        dz_real next_similarity = 0.0;
        dz_real context_similarity = 0.0;
        dz_real transition_similarity = 0.0;
        dz_real padic_similarity = 0.0;
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
        if (is_structural_token(t)) return false;
        if (t.size() > 22) return true;
        if (t.size() <= 1) return true;
        if (t == "\n") return true;
        if (t.find("toolu_") != std::string::npos) return true;
        int digs = 0;
        for (auto c : t) if (c >= '0' && c <= '9') digs++;
        if (digs > (int)t.size() * 2 / 3) return true;
        return false;
    }

    // Tokens the generator may emit: real words plus structural
    // punctuation/operators. Free-form single characters stay excluded.
    static bool generation_candidate_token(const std::string& token) {
        return token.size() > 1 || is_structural_token(token);
    }

    void embed(std::string_view text) {
        auto tokens = tokenize_code(text, std::min<std::size_t>(text.size(), 2048));
        embed_tokens(tokens);
    }

    void embed_tokens(const std::vector<std::string>& tokens) {
        for (const auto& t : tokens) {
            if (t.empty() || t == " " || t == "\t") continue;
            // Subword '##' traces are never emitted by generation and are no
            // longer trained (learn() operates in query-token space), so
            // storing an oscillator per '##' twin only doubled the oscillator
            // budget and the saved-model size.
            if (is_subword_continuation(t)) continue;
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

        // Training projections are built in QUERY-TOKEN SPACE: the same
        // lowercased alnum stream and the same multi-scale recency waves
        // that field_impulse_signature uses when forward() projects a
        // prompt. They used to be built from the raw tokenize_code stream
        // (case-sensitive, with '##' subword twins and punctuation shifting
        // every position index), which made learned keys statistically
        // independent of inference-time states: the measured cosine between
        // the two spaces for identical text was ~0.12 versus 1.0 after this
        // alignment. The wave kernel itself is translation-invariant (see
        // field_state.h), so a context learned at one line offset now
        // matches the identical context at any other offset.
        std::vector<std::string> query_runs;
        std::vector<std::size_t> query_prefix(train_tokens + 1, 0);
        query_runs.reserve(train_tokens);
        for (std::size_t i = 0; i < train_tokens; ++i) {
            query_prefix[i] = query_runs.size();
            if (is_subword_continuation(tokens[i])) {
                continue;
            }
            append_query_runs(tokens[i], query_runs);
        }
        query_prefix[train_tokens] = query_runs.size();
        const std::size_t query_count = query_runs.size();
        if (query_count == 0) {
            return;
        }

        struct FieldProjection {
            std::vector<cx> ampl;
            std::vector<dz_real> padic;
        };

        // Prefix signatures accumulate sequentially (cheap). The expensive
        // spectral transforms are independent per prefix: with enough
        // prefixes each one runs serially as its own work item, otherwise a
        // single prefix at a time parallelizes internally over dimensions.
        // Both paths produce bit-identical projections for any worker count
        // because the transform's dither restarts at fixed block boundaries.
        std::vector<FieldProjection> projections(query_count + 1);
        std::vector<std::vector<dz_real>> signatures(query_count);
        {
            // The SAME multi-scale kernel forward() consumes through
            // field_impulse_signature; snapshots after each pushed token
            // give the per-prefix signatures.
            FieldWaveAccumulator accumulator(seed_primes_.size());
            std::vector<long double> wave(seed_primes_.size(), 0.0L);
            std::vector<long double> snapshot;
            for (std::size_t len = 1; len <= query_count; ++len) {
                field_token_wave(query_runs[len - 1], seed_primes_.size(), wave.data());
                accumulator.push_wave(wave.data());
                accumulator.signature_into(snapshot);
                signatures[len - 1].assign(snapshot.begin(), snapshot.end());
            }
        }
        if (effective_thread_count(query_count) > 1) {
            parallel_for_ranges(query_count, [&](std::size_t begin, std::size_t end) {
                for (std::size_t index = begin; index < end; ++index) {
                    seed_weyl_transform_from_signature(signatures[index],
                                                       projections[index + 1].ampl,
                                                       projections[index + 1].padic,
                                                       "",
                                                       false,
                                                       false);
                }
            });
        } else {
            for (std::size_t len = 1; len <= query_count; ++len) {
                seed_weyl_transform_from_signature(signatures[len - 1],
                                                   projections[len].ampl,
                                                   projections[len].padic,
                                                   "",
                                                   false,
                                                   true);
            }
        }

        std::vector<cx> transition;
        std::vector<std::string> context_tokens;

        for (std::size_t ti = 0; ti < train_tokens; ++ti) {
            // Absolute query-space prefix lengths strictly before and after
            // this token. '##' traces contribute no query tokens and only
            // extend the lexical window; training them against a degenerate
            // context==current pair would pull unrelated oscillators toward
            // one shared point (the global-attractor failure mode).
            // STRUCTURAL tokens (punctuation/operators) also advance nothing
            // in query space, but they train deliberately as context-anchored
            // milestones: key == query == the context projection, so ':' is
            // retrievable exactly where it belongs while leaving the state
            // trajectory untouched.
            const std::size_t context_len = query_prefix[ti];
            const std::size_t current_len = query_prefix[ti + 1];
            const bool structural = is_structural_token(tokens[ti]);
            if (bad_token(tokens[ti]) || context_len == 0 ||
                (current_len == context_len && !structural)) {
                push_context_token(context_tokens, tokens[ti]);
                continue;
            }
            auto found = token_index_.find(tokens[ti]);
            if (found == token_index_.end()) {
                push_context_token(context_tokens, tokens[ti]);
                continue;
            }
            const auto positive_index = found->second;

            const std::size_t target_len = structural ? context_len : current_len;
            if (context_len >= projections.size() || target_len >= projections.size() ||
                projections[context_len].ampl.empty() || projections[target_len].ampl.empty()) {
                push_context_token(context_tokens, tokens[ti]);
                continue;
            }

            const auto& context_proj = projections[context_len];
            const auto& current_proj = projections[target_len];

            if (std::all_of(context_proj.ampl.begin(), context_proj.ampl.end(), [](cx v) {
                    return std::abs(v) < 1e-30;
                })) {
                push_context_token(context_tokens, tokens[ti]);
                continue;
            }

            if (update_probability_ < 1.0 && random_unit() > update_probability_) {
                push_context_token(context_tokens, tokens[ti]);
                continue;
            }

            std::vector<cx> ctx_ampl = context_proj.ampl;
            std::vector<cx> curr_ampl = current_proj.ampl;
            std::vector<dz_real> ctx_padic = context_proj.padic;
            std::vector<dz_real> curr_padic = current_proj.padic;

            if (update_noise_ > 0.0) {
                add_complex_noise(ctx_ampl, update_noise_);
                add_complex_noise(curr_ampl, update_noise_);
                add_real_noise(ctx_padic, update_noise_);
                add_real_noise(curr_padic, update_noise_);
            }

            spectral_bridge_into(ctx_ampl, curr_ampl, transition);
            const dz_real own_match = update_oscillator(oscs_[positive_index],
                                                        ctx_ampl,
                                                        curr_ampl,
                                                        transition,
                                                        ctx_padic,
                                                        curr_padic,
                                                        context_tail_hashes(context_tokens));
            update_contrastive_negatives(positive_index, ctx_ampl, ctx_padic, own_match);
            push_context_token(context_tokens, tokens[ti]);
        }
    }

    std::string forward(std::string_view text, std::size_t max_tokens = 24) {
        constexpr std::size_t no_prototype = std::numeric_limits<std::size_t>::max();
        auto [fp, current_padic] = seed_weyl_transform(text, true);
        std::string out;
        std::vector<std::string> recently_generated;
        // Long-tail repetition penalty table: the short-range formula returns
        // ~0.996 at distance 11, so period-11+ loops used to be nearly free.
        // pen(11)=0.45, pen(16)=0.68, pen(24)=0.87, pen(32)=0.95.
        dz_real long_pen[33] = {};
        for (std::size_t d = 11; d <= 32; ++d) {
            long_pen[d] = 1.0 - 0.55 * std::exp(-static_cast<dz_real>(d - 11) / 9.0);
        }
        struct SavedOscillator {
            std::size_t index;
            std::vector<cx> query;
            std::vector<cx> transition;
            std::vector<std::vector<cx>> proto_queries;
            std::vector<std::vector<cx>> proto_transitions;
        };
        std::vector<SavedOscillator> saved_oscs;
        auto save_oscillator = [&](std::size_t idx) {
            for (const auto& saved : saved_oscs) {
                if (saved.index == idx) return;
            }
            SavedOscillator saved;
            saved.index = idx;
            saved.query = oscs_[idx].query;
            saved.transition = oscs_[idx].transition;
            saved.proto_queries.reserve(oscs_[idx].prototypes.size());
            saved.proto_transitions.reserve(oscs_[idx].prototypes.size());
            for (const auto& proto : oscs_[idx].prototypes) {
                saved.proto_queries.push_back(proto.query);
                saved.proto_transitions.push_back(proto.transition);
            }
            saved_oscs.push_back(std::move(saved));
        };
        auto normalize = [&]() {
            dz_real n = 0.0;
            for (auto v : fp) n += complex_norm(v);
            dz_real fn = std::sqrt(n);
            if (fn > 1e-30) for (auto& v : fp) v /= fn;
        };
        normalize();
        inject_prompt_resonance(text, fp, current_padic);
        normalize();
        const std::vector<cx> prompt_trace = fp;
        const std::vector<dz_real> prompt_padic_trace = current_padic;
        std::vector<cx> attractor_center;
        std::vector<dz_real> attractor_padic_center;
        std::vector<std::vector<cx>> attractor_basis;
        std::vector<std::vector<dz_real>> attractor_padic_basis;
        // The corpus center is always built: the conditional-contrast drive
        // needs each candidate's center fit regardless of interference. The
        // deflation subspace stays interference-only.
        build_attractor_center(attractor_center, attractor_padic_center);
        if (dimension_interference_ > 0.0) {
            build_attractor_subspace(attractor_center,
                                     attractor_padic_center,
                                     attractor_basis,
                                     attractor_padic_basis);
        }
        std::vector<cx> prompt_delta = prompt_trace;
        std::vector<dz_real> prompt_padic_delta = prompt_padic_trace;
        if (dimension_interference_ > 0.0) {
            remove_attractor_projection(prompt_delta, attractor_center);
            remove_attractor_projection(prompt_padic_delta, attractor_padic_center);
            remove_attractor_subspace_projection(prompt_delta, attractor_basis);
            remove_attractor_subspace_projection(prompt_padic_delta, attractor_padic_basis);
        }
        struct PromptAxis {
            std::vector<cx> field;
            std::vector<dz_real> padic;
            dz_real weight = 1.0;
        };
        std::vector<PromptAxis> prompt_axes;
        if (dimension_interference_ > 0.0) {
            const auto prompt_tokens = tokenize_query(text);
            prompt_axes.reserve(prompt_tokens.size() * 2U + 1U);
            const auto add_prompt_axis = [&](std::string_view axis_text, dz_real weight) {
                PromptAxis axis;
                seed_weyl_transform_into(axis_text, axis.field, axis.padic, true, true);
                remove_attractor_projection(axis.field, attractor_center);
                remove_attractor_projection(axis.padic, attractor_padic_center);
                remove_attractor_subspace_projection(axis.field, attractor_basis);
                remove_attractor_subspace_projection(axis.padic, attractor_padic_basis);
                axis.weight = weight;
                prompt_axes.push_back(std::move(axis));
            };
            for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
                if (prompt_tokens[i].size() > 1) {
                    add_prompt_axis(prompt_tokens[i],
                                    std::clamp<dz_real>(static_cast<dz_real>(prompt_tokens[i].size()) / 6.0,
                                               0.65,
                                               1.65));
                }
                if (i > 0 && prompt_tokens[i - 1].size() > 1 && prompt_tokens[i].size() > 1) {
                    add_prompt_axis(prompt_tokens[i - 1] + " " + prompt_tokens[i], 1.35);
                }
            }
            add_prompt_axis(text, 1.0);
        }
        std::vector<cx> prompt_anchor_field;
        std::vector<dz_real> prompt_anchor_padic;
        if (dimension_interference_ > 0.0) {
            build_prompt_anchor_field(text,
                                      attractor_center,
                                      attractor_padic_center,
                                      attractor_basis,
                                      attractor_padic_basis,
                                      prompt_anchor_field,
                                      prompt_anchor_padic);
        }
        std::vector<std::uint64_t> active_tail = lexical_tail(text);
        const auto push_active_token = [&](const std::string& token) {
            // One hash per lowercased alnum run — the same convention
            // lexical_tail and the trained context tails use; hashing the
            // raw token meant capitalized or multi-run output never matched
            // any trained tail entry.
            std::vector<std::string> runs;
            append_query_runs(token, runs);
            for (const auto& run : runs) {
                if (run.size() <= 1) {
                    continue;
                }
                active_tail.push_back(stable_hash(run));
            }
            constexpr std::size_t tail_limit = 6;
            if (active_tail.size() > tail_limit) {
                active_tail.erase(active_tail.begin(), active_tail.end() - tail_limit);
            }
        };

        std::size_t previous_oscillator_index = std::numeric_limits<std::size_t>::max();
        std::size_t previous_prototype_index = no_prototype;
        std::vector<cx> previous_fp = fp;

        struct Candidate {
            dz_real score;
            std::size_t oscillator;
            std::size_t prototype;
        };
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
        const auto candidate_padic = [&](const Candidate& candidate) -> const std::vector<dz_real>& {
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
        const auto candidate_negative_key = [&](const Candidate& candidate) -> const std::vector<cx>& {
            const auto& oscillator = oscs_[candidate.oscillator];
            if (candidate.prototype != no_prototype && candidate.prototype < oscillator.prototypes.size()) {
                return oscillator.prototypes[candidate.prototype].negative_key;
            }
            return oscillator.negative_key;
        };
        const auto candidate_negative_padic = [&](const Candidate& candidate) -> const std::vector<dz_real>& {
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

        // ---- Step-invariant candidate statics ----------------------------
        // Keys, p-adic signatures, negative memories, strength/error and the
        // whole prompt/attractor geometry stay fixed for the entire
        // generation loop (fast weights only touch query/transition of the
        // previous winner), so every fit term that depends only on them is
        // computed once here instead of once per emitted token. Each hoisted
        // expression is written exactly as the old inline one, so the scores
        // are bit-identical; only the moving-state terms remain in the
        // per-step loop.
        const bool has_interference = dimension_interference_ > 0.0;
        const dz_real dd_coeff = 0.16 + 1.85 * dimension_interference_;
        const dz_real sens_coeff = 7.0 * dimension_interference_;
        // Contrast strength ramps with interference so the di=0 baseline
        // stays a pure-drive path (prompt_deflation's collapse contract).
        const dz_real beta_eff =
            contrast_beta_ * std::clamp<dz_real>(dimension_interference_ / 0.25, 0.0, 1.0);
        struct CandidateStatic {
            std::size_t oscillator = 0;
            std::size_t prototype = 0;
            dz_real padic_norm = 0.0;
            dz_real negative_key_norm = 0.0;
            dz_real negative_padic_norm = 0.0;
            dz_real reliability = 0.0;
            dz_real prompt_specificity = 0.0;
            dz_real differential_drive = 0.0;
            dz_real route_gain = 1.0;
            dz_real anti_template = 1.0;
            dz_real subspace_penalty = 1.0;
            dz_real attractor_pressure = 0.0;
            dz_real delta_fit = 0.0;
            dz_real anchor_fit = 0.0;
            dz_real axis_best = 0.0;
            dz_real anchor_mix84 = 0.0;  // 0.84*anchor_fit + 0.16*anchor_padic_fit
            dz_real anchor_mix82 = 0.0;  // 0.82*anchor_fit + 0.18*anchor_padic_fit
            dz_real delta_mix68 = 0.0;   // 0.68*delta_fit + 0.32*delta_padic_fit
            dz_real prompt_mix = 0.0;    // 0.45*prompt_fit + 0.25*prompt_padic_fit
            dz_real center_fit = 0.0;    // |<key, corpus center>| for the contrast drive
        };
        std::vector<CandidateStatic> statics;
        statics.reserve(oscs_.size());
        for (std::size_t i = 0; i < oscs_.size(); ++i) {
            if (!generation_candidate_token(oscs_[i].token)) continue;
            if (is_subword_continuation(oscs_[i].token)) continue;
            const std::size_t prototypes = std::max<std::size_t>(1, oscs_[i].prototypes.size());
            for (std::size_t p = 0; p < prototypes; ++p) {
                CandidateStatic entry;
                entry.oscillator = i;
                entry.prototype = oscs_[i].prototypes.empty() ? no_prototype : p;
                statics.push_back(entry);
            }
        }
        const auto fill_static_range = [&](std::size_t begin, std::size_t end) {
            for (std::size_t k = begin; k < end; ++k) {
                auto& cs = statics[k];
                const Candidate candidate{0.0, cs.oscillator, cs.prototype};
                const auto& key = candidate_key(candidate);
                const auto& padic = candidate_padic(candidate);
                const auto& negative_key = candidate_negative_key(candidate);
                const auto& negative_padic_sig = candidate_negative_padic(candidate);
                dz_real padic_norm_value = 0.0;
                for (std::size_t j = 0; j < padic.size(); ++j) {
                    padic_norm_value += padic[j] * padic[j];
                }
                cs.padic_norm = padic_norm_value;
                dz_real negative_key_norm = 0.0;
                for (std::size_t j = 0; j < negative_key.size(); ++j) {
                    negative_key_norm += complex_norm(negative_key[j]);
                }
                cs.negative_key_norm = negative_key_norm;
                dz_real negative_padic_norm = 0.0;
                for (std::size_t j = 0; j < negative_padic_sig.size(); ++j) {
                    negative_padic_norm += negative_padic_sig[j] * negative_padic_sig[j];
                }
                cs.negative_padic_norm = negative_padic_norm;
                const auto& oscillator = oscs_[cs.oscillator];
                const dz_real observations =
                    static_cast<dz_real>(std::max<std::size_t>(1, oscillator.observations));
                // The conditional-contrast drive replaces most of the raw
                // frequency punishment: pressure drops from 0.75*di to
                // frequency_pressure_scale_*di, and the center subtraction in
                // the per-step drive handles the fits-everywhere templates.
                // Half-PMI prior replaces the old frequency_penalty and
                // content_gain: a weak sqrt-count prior keeps runaway tokens
                // in check while the conditional-contrast drive handles
                // fits-everywhere templates. A 600-observation keyword now
                // keeps ~0.67 of its score instead of ~0.13 — the measured
                // syntax-backbone deletion on code corpora. Structural tokens
                // are fully exempt: syntax glue is frequent BECAUSE it is
                // structural.
                const bool structural_candidate = is_structural_token(oscillator.token);
                const dz_real count_prior =
                    structural_candidate
                        ? 1.0
                        : 1.0 / (1.0 + 0.02 * std::sqrt(observations));
                cs.reliability = oscillator.strength * count_prior / (1.0 + oscillator.error_ema);
                cs.center_fit =
                    !attractor_center.empty() ? normalized_complex_similarity(attractor_center, key) : 0.0;
                const dz_real prompt_fit =
                    has_interference ? normalized_complex_similarity(prompt_trace, key) : 0.0;
                const dz_real prompt_padic_fit =
                    has_interference
                        ? std::max<dz_real>(0.0, normalized_cosine(prompt_padic_trace, padic))
                        : 0.0;
                const dz_real delta_fit =
                    has_interference && !prompt_delta.empty()
                        ? normalized_complex_similarity(prompt_delta, key)
                        : 0.0;
                const dz_real delta_padic_fit =
                    has_interference && !prompt_padic_delta.empty()
                        ? std::max<dz_real>(0.0, normalized_cosine(prompt_padic_delta, padic))
                        : 0.0;
                const dz_real attractor_fit = has_interference ? cs.center_fit : 0.0;
                const dz_real attractor_padic_fit =
                    has_interference && !attractor_padic_center.empty()
                        ? 0.5 + 0.5 * normalized_cosine(attractor_padic_center, padic)
                        : 0.0;
                const dz_real attractor_mode_fit =
                    has_interference && !attractor_basis.empty()
                        ? complex_subspace_pressure(key, attractor_basis)
                        : 0.0;
                const dz_real attractor_padic_mode_fit =
                    has_interference && !attractor_padic_basis.empty()
                        ? real_subspace_pressure(padic, attractor_padic_basis)
                        : 0.0;
                const dz_real attractor_pressure =
                    std::clamp<dz_real>(0.42 * attractor_fit + 0.16 * attractor_padic_fit +
                                   0.31 * attractor_mode_fit + 0.11 * attractor_padic_mode_fit,
                               0.0,
                               1.0);
                const dz_real anchor_fit =
                    has_interference && !prompt_anchor_field.empty()
                        ? normalized_complex_similarity(prompt_anchor_field, key)
                        : 0.0;
                const dz_real anchor_padic_fit =
                    has_interference && !prompt_anchor_padic.empty()
                        ? std::max<dz_real>(0.0, normalized_cosine(prompt_anchor_padic, padic))
                        : 0.0;
                dz_real axis_best = 0.0;
                dz_real axis_mean = 0.0;
                if (has_interference && !prompt_axes.empty()) {
                    for (const auto& axis : prompt_axes) {
                        const dz_real axis_field = normalized_complex_similarity(axis.field, key);
                        const dz_real axis_padic =
                            std::max<dz_real>(0.0, normalized_cosine(axis.padic, padic));
                        const dz_real axis_score = axis.weight * (0.88 * axis_field + 0.12 * axis_padic);
                        axis_best = std::max(axis_best, axis_score);
                        axis_mean += axis_score;
                    }
                    axis_mean /= static_cast<dz_real>(prompt_axes.size());
                }
                const dz_real axis_drive =
                    has_interference
                        ? std::max<dz_real>(0.0,
                                                axis_best - 0.72 * axis_mean - 0.42 * attractor_pressure)
                        : 0.0;
                cs.prompt_specificity =
                    std::max<dz_real>(0.0,
                                          0.40 * prompt_fit + 0.16 * prompt_padic_fit +
                                              0.28 * delta_fit + 0.08 * delta_padic_fit +
                                              0.34 * axis_drive +
                                              0.32 * anchor_fit + 0.10 * anchor_padic_fit -
                                              0.62 * attractor_fit - 0.18 * attractor_padic_fit);
                cs.delta_mix68 = 0.68 * delta_fit + 0.32 * delta_padic_fit;
                cs.differential_drive =
                    has_interference
                        ? std::max<dz_real>(0.0,
                                                std::max(cs.delta_mix68, axis_drive) -
                                                    0.35 * attractor_pressure)
                        : 0.0;
                cs.route_gain =
                    has_interference
                        ? std::clamp<dz_real>(0.25 + 26.0 * dimension_interference_ * cs.prompt_specificity,
                                     0.30,
                                     4.60)
                        : 1.0;
                // Softened: the conditional-contrast drive is now the primary
                // owner of center suppression; anti_template only trims.
                cs.anti_template =
                    has_interference
                        ? std::clamp<dz_real>(1.18 - (0.40 + 0.80 * dimension_interference_) * attractor_pressure,
                                     0.45,
                                     1.18)
                        : 1.0;
                cs.subspace_penalty =
                    has_interference
                        ? std::clamp<dz_real>(1.0 -
                                         (0.86 + 2.85 * dimension_interference_) * attractor_mode_fit -
                                         (0.16 + 1.05 * dimension_interference_) *
                                             attractor_padic_mode_fit,
                                     0.25,
                                     1.0)
                        : 1.0;
                cs.attractor_pressure = attractor_pressure;
                cs.delta_fit = delta_fit;
                cs.anchor_fit = anchor_fit;
                cs.axis_best = axis_best;
                cs.anchor_mix84 = 0.84 * anchor_fit + 0.16 * anchor_padic_fit;
                cs.anchor_mix82 = 0.82 * anchor_fit + 0.18 * anchor_padic_fit;
                cs.prompt_mix = 0.45 * prompt_fit + 0.25 * prompt_padic_fit;
            }
        };
        const bool parallel_candidates =
            thread_count_ > 1 && dim_ >= parallel_min_dimensions_ &&
            statics.size() >= thread_count_ * 2U;
        if (parallel_candidates) {
            const std::size_t workers = std::min(thread_count_, statics.size());
            range_pool(workers).run(statics.size(), workers, fill_static_range);
        } else {
            fill_static_range(0, statics.size());
        }
        std::vector<dz_real> candidate_scores(statics.size(), -1.0);
        std::vector<Candidate> raw;
        raw.reserve(statics.size());

        for (std::size_t s = 0; s < max_tokens; ++s) {
            parallel_for_ranges(dim_, [&](std::size_t begin, std::size_t end) {
                for (std::size_t z = begin; z < end; ++z) {
                    dz_real theta = static_cast<dz_real>(s) * 0.005 * (1.0 + static_cast<dz_real>(z) * 0.1);
                    dz_real st = 0.0;
                    dz_real ct = 1.0;
                    sincos_ld(theta, st, ct);
                    fp[z] *= cx(ct, st);
                }
            });
            std::vector<cx> routed_fp = fp;
            std::vector<dz_real> routed_padic = current_padic;
            std::vector<cx> counter_routed_fp = fp;
            if (dimension_interference_ > 0.0) {
                apply_prompt_hamiltonian_transport(routed_fp,
                                                   routed_padic,
                                                   prompt_delta,
                                                   prompt_padic_delta,
                                                   s,
                                                   1.0);
                std::vector<dz_real> counter_padic = current_padic;
                apply_prompt_hamiltonian_transport(counter_routed_fp,
                                                   counter_padic,
                                                   prompt_delta,
                                                   prompt_padic_delta,
                                                   s,
                                                   -1.0);
            }

            // ---- Per-step candidate scoring ------------------------------
            // Only the moving-state terms are computed here; everything else
            // comes from the precomputed statics. transported_fit reuses the
            // dm dot product (normalized_complex_similarity(routed_fp, key)
            // equals |dm| by conjugate symmetry), and the cached-norm
            // similarity helpers skip the O(dim) negative-memory passes
            // entirely while the negative vectors are still all-zero.
            dz_real routed_fp_norm = 0.0;
            for (std::size_t j = 0; j < dim_; ++j) {
                routed_fp_norm += complex_norm(routed_fp[j]);
            }
            dz_real routed_padic_norm = 0.0;
            for (std::size_t j = 0; j < routed_padic.size(); ++j) {
                routed_padic_norm += routed_padic[j] * routed_padic[j];
            }
            // Periodic-run lengths for bigram/trigram cycle penalties:
            // run_p = longest consecutive suffix of the window satisfying
            // W[n-i] == W[n-i-p]. Computed once per step, read-only in the
            // scoring workers, so parallel bit-identity is unaffected.
            const std::size_t rec_n = recently_generated.size();
            const auto periodic_run = [&](std::size_t period) {
                std::size_t run = 0;
                for (std::size_t i = 1; i + period <= rec_n; ++i) {
                    if (recently_generated[rec_n - i] == recently_generated[rec_n - i - period]) {
                        ++run;
                    } else {
                        break;
                    }
                }
                return run;
            };
            const std::size_t run2 = periodic_run(2);
            const std::size_t run3 = periodic_run(3);
            const std::string* cycle2_token = rec_n >= 2 ? &recently_generated[rec_n - 2] : nullptr;
            const std::string* cycle3_token = rec_n >= 3 ? &recently_generated[rec_n - 3] : nullptr;
            const auto score_range = [&](std::size_t begin, std::size_t end) {
                for (std::size_t k = begin; k < end; ++k) {
                    const auto& cs = statics[k];
                    candidate_scores[k] = -1.0;
                    const auto& token = oscs_[cs.oscillator].token;
                    // strict repetition ban for the last 10 tokens (except short words <= 3 chars)
                    bool strictly_banned = false;
                    dz_real repetition_penalty = 1.0;
                    if (!recently_generated.empty()) {
                        for (std::size_t r = 0; r < recently_generated.size(); ++r) {
                            if (recently_generated[r] == token) {
                                const std::size_t distance = recently_generated.size() - r;
                                if (distance <= 10 && token.size() > 3) {
                                    strictly_banned = true;
                                    break;
                                }
                                const dz_real dist_ld = static_cast<dz_real>(distance);
                                // Short tokens keep the smooth formula at all
                                // distances (they are never hard-banned, and
                                // switching them to the long tail would make
                                // an OLDER occurrence more punished than a
                                // recent one across the 10 -> 11 seam). Long
                                // tokens are banned through distance 10, so
                                // for them the long tail starts monotonically
                                // at 0.45.
                                const dz_real penalty =
                                    distance <= 10 || token.size() <= 3
                                        ? 0.08 + 0.92 * (1.0 - 1.0 / dist_ld)
                                        : long_pen[distance];
                                repetition_penalty = std::min(repetition_penalty, penalty);
                            }
                        }
                    }
                    if (strictly_banned) continue;
                    // Bigram/trigram cycle damping: fires only when at least
                    // one full period already exists in the window AND this
                    // candidate would extend it. Never zeroes a score. By
                    // construction this only reaches tokens of <= 3 chars —
                    // longer tokens at distance 2 or 3 are already hard
                    // banned above — so it specifically damps short-word
                    // cycles like "of the of the".
                    dz_real cycle_penalty = 1.0;
                    // Structural tokens are exempt from the frequency prior,
                    // so their cycles need a much stiffer spring: "* += * +="
                    // ping-pong otherwise rides the exemption.
                    const bool structural_token = is_structural_token(token);
                    const dz_real cycle_beta = structural_token ? 1.20 : 0.35;
                    if (structural_token && rec_n > 0) {
                        // Occupancy guard: broken ping-pong ("* += * ord +=")
                        // dodges the consecutive-cycle detector, so any
                        // structural token flooding the recent window decays
                        // geometrically past 3 occurrences in the last 12.
                        const std::size_t occ_window = std::min<std::size_t>(rec_n, 12);
                        std::size_t recent_count = 0;
                        for (std::size_t r = rec_n - occ_window; r < rec_n; ++r) {
                            if (recently_generated[r] == token) {
                                ++recent_count;
                            }
                        }
                        if (recent_count > 3) {
                            cycle_penalty *= std::pow(static_cast<dz_real>(0.5),
                                                      static_cast<dz_real>(recent_count - 3));
                        }
                    }
                    if (cycle2_token != nullptr && run2 >= 2 && token == *cycle2_token) {
                        const dz_real cycles = (static_cast<dz_real>(run2) + 1.0) / 2.0;
                        cycle_penalty /= 1.0 + cycle_beta * cycles * cycles;
                    }
                    if (cycle3_token != nullptr && run3 >= 3 && token == *cycle3_token) {
                        const dz_real cycles = (static_cast<dz_real>(run3) + 1.0) / 3.0;
                        cycle_penalty /= 1.0 + cycle_beta * cycles * cycles;
                    }
                    const dz_real rep_total =
                        std::max(repetition_penalty * cycle_penalty, static_cast<dz_real>(1.0e-4));
                    const Candidate candidate{0.0, cs.oscillator, cs.prototype};
                    const auto& key = candidate_key(candidate);
                    cx dm = 0;
                    for (std::size_t j = 0; j < dim_; ++j) {
                        dm += conjugate_multiply(key[j], routed_fp[j]);
                    }
                    const dz_real abs_dm = static_cast<dz_real>(std::abs(dm));
                    const dz_real padic_match =
                        0.5 + 0.5 * cosine_cached(routed_padic,
                                                    candidate_padic(candidate),
                                                    routed_padic_norm,
                                                    cs.padic_norm);
                    const dz_real lexical_match = tail_overlap(active_tail, candidate_tail(candidate));
                    const dz_real raw_state_fit =
                        has_interference ? normalized_complex_similarity(counter_routed_fp, key) : 0.0;
                    const dz_real transported_fit =
                        has_interference ? std::clamp<dz_real>(abs_dm, 0.0, 1.0) : 0.0;
                    const dz_real differential_sensitivity =
                        has_interference
                            ? std::max<dz_real>(0.0,
                                                    transported_fit - raw_state_fit +
                                                        0.22 * cs.delta_fit - 0.24 * cs.attractor_pressure)
                            : 0.0;
                    // Conditional-contrast drive (half-PMI): the candidate's
                    // raw fit is discounted multiplicatively by its fit to
                    // the global corpus center, so a token wins by fitting
                    // THIS context better than contexts in general.
                    // Frequent-but-specific syntax survives;
                    // fits-everywhere template words do not. Ramped off at
                    // di=0 to keep the baseline path pure.
                    const dz_real conditional_drive =
                        has_interference
                            ? abs_dm * std::max<dz_real>(1.0 - beta_eff * cs.center_fit,
                                                         contrast_floor_)
                            : abs_dm;
                    const dz_real field_drive =
                        has_interference
                            ? std::max<dz_real>(conditional_drive, dd_coeff * cs.differential_drive)
                            : conditional_drive;
                    const dz_real context_gate =
                        has_interference
                            ? std::clamp<dz_real>(0.10 + 1.50 * lexical_match + 1.15 * cs.prompt_specificity,
                                         0.30,
                                         2.10)
                            : 0.40 + 0.90 * lexical_match;
                    const dz_real bridge_fit =
                        projected_bridge_similarity(routed_fp,
                                                    candidate_transition(candidate),
                                                    candidate_query(candidate));
                    const dz_real negative_spectral =
                        complex_similarity_cached(routed_fp,
                                                  candidate_negative_key(candidate),
                                                  routed_fp_norm,
                                                  cs.negative_key_norm);
                    const dz_real negative_padic =
                        0.5 + 0.5 * cosine_cached(routed_padic,
                                                    candidate_negative_padic(candidate),
                                                    routed_padic_norm,
                                                    cs.negative_padic_norm);
                    const dz_real collision =
                        std::clamp<dz_real>(0.78 * negative_spectral + 0.22 * negative_padic, 0.0, 1.0);
                    const dz_real contrastive_penalty =
                        std::clamp<dz_real>(1.0 - contrastive_strength_ * collision, 0.12, 1.0);
                    const dz_real sensitivity_gain =
                        has_interference
                            ? std::clamp<dz_real>(0.58 + sens_coeff * differential_sensitivity, 0.30, 2.35)
                            : 1.0;
                    const dz_real prompt_axis_signal =
                        has_interference
                            ? std::max({cs.axis_best,
                                        cs.anchor_mix82,
                                        cs.delta_mix68,
                                        cs.prompt_mix + 0.30 * lexical_match})
                            : 1.0;
                    const dz_real prompt_axis_gate =
                        has_interference
                            ? std::clamp<dz_real>(0.06 + 1.85 * prompt_axis_signal +
                                             0.55 * differential_sensitivity -
                                             0.38 * cs.attractor_pressure,
                                         0.20,
                                         1.75)
                            : 1.0;
                    const dz_real anchor_gate =
                        has_interference && !prompt_anchor_field.empty()
                            ? std::clamp<dz_real>(0.05 + 2.15 * cs.anchor_mix84 +
                                             0.35 * differential_sensitivity -
                                             0.22 * cs.attractor_pressure,
                                         0.20,
                                         1.85)
                            : 1.0;
                    const dz_real context_specificity_gate =
                        has_interference
                            ? std::clamp<dz_real>(0.08 + 3.20 * lexical_match +
                                             0.72 * differential_sensitivity +
                                             0.34 * cs.anchor_fit -
                                             0.18 * cs.attractor_pressure,
                                         0.20,
                                         1.85)
                            : 1.0;
                    candidate_scores[k] =
                        field_drive * (0.70 + 0.30 * padic_match) *
                        (0.75 + 0.25 * bridge_fit) * cs.reliability * context_gate *
                        contrastive_penalty * cs.route_gain * cs.anti_template *
                        sensitivity_gain * cs.subspace_penalty * prompt_axis_gate *
                        anchor_gate * context_specificity_gate * rep_total;
                }
            };
            if (parallel_candidates) {
                const std::size_t workers = std::min(thread_count_, statics.size());
                range_pool(workers).run(statics.size(), workers, score_range);
            } else {
                score_range(0, statics.size());
            }
            raw.clear();
            for (std::size_t k = 0; k < statics.size(); ++k) {
                if (candidate_scores[k] > 1e-12) {
                    raw.push_back({candidate_scores[k], statics[k].oscillator, statics[k].prototype});
                }
            }
            if (raw.empty()) break;
            const auto by_score = [](const Candidate& left, const Candidate& right) {
                return left.score > right.score;
            };
            std::partial_sort(raw.begin(), raw.begin() + std::min<std::size_t>(64, raw.size()),
                              raw.end(), by_score);
            // Feynman Path Integral Rollout (3-step mental lookahead in phase space)
            if (raw.size() > 1) {
                const std::size_t rollout_candidates = std::min<std::size_t>(16, raw.size());
                // Successor states are searched among the top scored
                // candidates of this step. The old scan took the first
                // min(48, oscs_.size()) oscillators in insertion order — an
                // arbitrary early-corpus subset (silently reshuffled further
                // by drop_one eviction swaps), so lookahead rewarded
                // resonance with whatever happened to be embedded first.
                const std::size_t successor_limit = std::min<std::size_t>(48, raw.size());
                const auto rollout_one = [&](std::size_t c) {
                    const Candidate candidate = raw[c];
                    dz_real path_action = 0.0;
                    dz_real discount = 1.0;

                    std::vector<cx> sim_fp = fp;
                    const auto& query_1 = candidate_query(candidate);
                    const auto bridged_1 = apply_bridge(sim_fp, candidate_transition(candidate));

                    for (std::size_t j = 0; j < dim_; ++j) {
                        const cx trace_j = prompt_trace.empty() ? cx(0, 0) : prompt_trace[j];
                        sim_fp[j] = dz_real(0.38) * query_1[j] + dz_real(0.40) * bridged_1[j] + dz_real(0.22) * trace_j;
                    }
                    if (dimension_interference_ > 0.0) {
                        apply_dimensional_interference(oscs_[candidate.oscillator].token, oscs_[candidate.oscillator].padic_signature, sim_fp);
                    }
                    normalize_complex(sim_fp);

                    // Ban-consistent lookahead: the rollout used to reward
                    // continuations whose tokens the real scorer would ban,
                    // inflating loop-shaped candidates.
                    const std::string* sim_hist[3] = {&oscs_[candidate.oscillator].token, nullptr, nullptr};
                    std::size_t sim_count = 1;
                    for (std::size_t h = 0; h < 2; ++h) {
                        dz_real best_next_score = 0.0;
                        std::size_t best_next = 0;
                        bool have_filtered = false;
                        dz_real best_any_score = 0.0;
                        std::size_t best_any = 0;

                        for (std::size_t t = 0; t < successor_limit; ++t) {
                            const Candidate next_cand{0.0, raw[t].oscillator, raw[t].prototype};
                            const auto& next_key = candidate_key(next_cand);
                            cx dm = 0;
                            for (std::size_t j = 0; j < dim_; ++j) {
                                dm += conjugate_multiply(sim_fp[j], next_key[j]);
                            }
                            const dz_real dm_val = static_cast<dz_real>(std::abs(dm));
                            if (dm_val > best_any_score) {
                                best_any_score = dm_val;
                                best_any = t;
                            }
                            const auto& successor_token = oscs_[raw[t].oscillator].token;
                            bool repeats = false;
                            for (std::size_t si = 0; si < sim_count; ++si) {
                                if (successor_token == *sim_hist[si]) {
                                    repeats = true;
                                    break;
                                }
                            }
                            if (!repeats) {
                                const std::size_t horizon = 9 - h;
                                for (std::size_t j = 1; j <= horizon && j <= rec_n; ++j) {
                                    if (successor_token == recently_generated[rec_n - j]) {
                                        repeats = true;
                                        break;
                                    }
                                }
                            }
                            dz_real scored = dm_val;
                            if (repeats) {
                                if (successor_token.size() > 3) {
                                    continue;  // mirror the real scorer's hard ban
                                }
                                scored *= 0.5;
                            }
                            if (scored > best_next_score) {
                                best_next_score = scored;
                                best_next = t;
                                have_filtered = true;
                            }
                        }
                        if (!have_filtered) {
                            // Every successor was a banned repeat: keep the
                            // best one so the simulation can continue, but at
                            // the same halved credit short repeats get.
                            best_next = best_any;
                            best_next_score = best_any_score * 0.5;
                        }

                        const Candidate next_best{0.0, raw[best_next].oscillator, raw[best_next].prototype};
                        const auto& query_next = candidate_query(next_best);
                        const auto bridged_next = apply_bridge(sim_fp, candidate_transition(next_best));

                        for (std::size_t j = 0; j < dim_; ++j) {
                            const cx trace_j = prompt_trace.empty() ? cx(0, 0) : prompt_trace[j];
                            sim_fp[j] = dz_real(0.38) * query_next[j] + dz_real(0.40) * bridged_next[j] + dz_real(0.22) * trace_j;
                        }
                        if (dimension_interference_ > 0.0) {
                            apply_dimensional_interference(oscs_[next_best.oscillator].token, oscs_[next_best.oscillator].padic_signature, sim_fp);
                        }
                        normalize_complex(sim_fp);
                        if (sim_count < 3) {
                            sim_hist[sim_count++] = &oscs_[next_best.oscillator].token;
                        }

                        discount *= 0.70;
                        path_action += discount * best_next_score;
                    }
                    const dz_real path_gate = 0.45 + 1.25 * path_action;
                    raw[c].score = candidate.score * path_gate;
                };
                // Rollout candidates only read .oscillator/.prototype of the
                // shared top entries and each writes its own .score, so they
                // run concurrently without ordering effects.
                if (thread_count_ > 1 && dim_ >= parallel_min_dimensions_ && rollout_candidates > 1) {
                    const std::size_t workers = std::min(thread_count_, rollout_candidates);
                    range_pool(workers).run(rollout_candidates, workers, [&](std::size_t begin, std::size_t end) {
                        for (std::size_t c = begin; c < end; ++c) {
                            rollout_one(c);
                        }
                    });
                } else {
                    for (std::size_t c = 0; c < rollout_candidates; ++c) {
                        rollout_one(c);
                    }
                }

                // Re-sort after rescoring candidates via path action
                std::partial_sort(raw.begin(), raw.begin() + std::min<std::size_t>(64, raw.size()),
                                  raw.end(), by_score);
            }
            std::size_t K = std::min<std::size_t>(64, raw.size());
            // lateral inhibition: suppress similar oscillators
            // Query norms are hoisted out of the O(K^2) pair loop; they were
            // recomputed inside every pair's dim-loop.
            std::vector<const std::vector<cx>*> top_queries(K);
            std::vector<dz_real> top_query_norms(K);
            for (std::size_t t = 0; t < K; ++t) {
                top_queries[t] = &candidate_query(raw[t]);
                dz_real norm = 0.0;
                for (std::size_t j = 0; j < dim_; ++j) {
                    norm += complex_norm((*top_queries[t])[j]);
                }
                top_query_norms[t] = norm;
            }
            // Each candidate's inhibition reads only the pre-inhibition raw
            // scores, so rows are independent and parallelize cleanly.
            std::vector<Candidate> inhibited(K);
            const auto inhibit_range = [&](std::size_t begin, std::size_t end) {
                for (std::size_t t = begin; t < end; ++t) {
                    Candidate candidate = raw[t];
                    dz_real score = candidate.score;
                    const auto& q = *top_queries[t];
                    for (std::size_t u = 0; u < t; ++u) {
                        const auto& qu = *top_queries[u];
                        cx cross = 0;
                        for (std::size_t j = 0; j < dim_; ++j) {
                            cross += conjugate_multiply(q[j], qu[j]);
                        }
                        const dz_real n1 = top_query_norms[t];
                        const dz_real n2 = top_query_norms[u];
                        dz_real sim = std::max<dz_real>(0.0, cross.real()) / (std::sqrt(n1 * n2) + 1e-30);
                        score -= 0.3 * sim * candidate.score;  // inhibition by similarity
                    }
                    candidate.score = score;
                    inhibited[t] = candidate;
                }
            };
            if (thread_count_ > 1 && dim_ >= parallel_min_dimensions_ && K >= 8) {
                const std::size_t workers = std::min(thread_count_, K);
                range_pool(workers).run(K, workers, inhibit_range);
            } else {
                inhibit_range(0, K);
            }
            if (generation_temperature_ > 0.01 && inhibited.size() > 1) {
                const std::size_t sample_k = std::min<std::size_t>(16, inhibited.size());
                std::vector<dz_real> probs(sample_k);
                for (std::size_t i = 0; i < sample_k; ++i) {
                    dz_real logit = std::log(std::max<dz_real>(1e-15, inhibited[i].score));
                    probs[i] = std::exp(logit / generation_temperature_);
                }
                dz_real sum_p = 0.0;
                for (auto p : probs) sum_p += p;
                
                // Sample from the field's seeded generator: the previous
                // thread_local random_device generator made forward()
                // nondeterministic at any temperature > 0.01 even with an
                // explicit constructor seed.
                std::uniform_real_distribution<dz_real> dist(0.0, sum_p);
                dz_real r = dist(rng_);
                dz_real running_sum = 0.0;
                std::size_t chosen_idx = 0;
                for (std::size_t i = 0; i < sample_k; ++i) {
                    running_sum += probs[i];
                    if (r <= running_sum) {
                        chosen_idx = i;
                        break;
                    }
                }
                if (chosen_idx > 0) {
                    std::swap(inhibited[0], inhibited[chosen_idx]);
                }
            } else {
                std::partial_sort(inhibited.begin(), inhibited.begin() + 1, inhibited.end(), by_score);
            }
            const Candidate best = inhibited[0];
            std::size_t best_i = best.oscillator;
            if (s > 0 && previous_oscillator_index != std::numeric_limits<std::size_t>::max()) {
                save_oscillator(previous_oscillator_index);
                auto& prev_osc = oscs_[previous_oscillator_index];
                const dz_real fast_rate = 0.20;
                const auto& current_key = candidate_key(best);
                std::vector<cx> trans_step;
                spectral_bridge_into(previous_fp, current_key, trans_step);
                for (std::size_t j = 0; j < dim_; ++j) {
                    prev_osc.transition[j] += fast_rate * (trans_step[j] - prev_osc.transition[j]);
                    prev_osc.query[j] += fast_rate * (current_key[j] - prev_osc.query[j]);
                }
                normalize_complex(prev_osc.transition);
                normalize_complex(prev_osc.query);
                if (previous_prototype_index != no_prototype &&
                    previous_prototype_index < prev_osc.prototypes.size()) {
                    auto& proto = prev_osc.prototypes[previous_prototype_index];
                    for (std::size_t j = 0; j < dim_; ++j) {
                        proto.transition[j] += fast_rate * (trans_step[j] - proto.transition[j]);
                        proto.query[j] += fast_rate * (current_key[j] - proto.query[j]);
                    }
                    normalize_complex(proto.transition);
                    normalize_complex(proto.query);
                }

                // Kuramoto-Adler Phase Synchronization (Phase-Locked Loop to prompt)
                const dz_real eta = 0.14; // coupling strength to prompt
                parallel_for_ranges(dim_, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t j = begin; j < end; ++j) {
                        const cx coupling = prompt_trace[j] * std::conj(previous_fp[j] * prev_osc.transition[j]);
                        const dz_real diff = std::atan2(coupling.imag(), coupling.real());
                        dz_real st = 0.0, ct = 1.0;
                        sincos_ld(eta * diff, st, ct);
                        prev_osc.transition[j] *= cx(ct, st);
                    }
                });

                // Josephson Junction Phase-Locked Current (coupling to actual state phase velocity)
                const dz_real g = 0.18; // Josephson coupling strength
                parallel_for_ranges(dim_, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t j = begin; j < end; ++j) {
                        const cx state_delta = fp[j] * std::conj(previous_fp[j]);
                        const dz_real delta_phase = std::atan2(state_delta.imag(), state_delta.real());
                        const dz_real trans_phase = std::arg(prev_osc.transition[j]);
                        const dz_real diff = delta_phase - trans_phase;
                        dz_real st = 0.0, ct = 1.0;
                        sincos_ld(g * std::sin(diff), st, ct);
                        prev_osc.transition[j] *= cx(ct, st);
                    }
                });
                normalize_complex(prev_osc.transition);
            }
            previous_oscillator_index = best_i;
            previous_prototype_index = best.prototype;
            previous_fp = fp;

            if (!is_subword_continuation(oscs_[best_i].token)) {
                const auto surface = subword_surface(oscs_[best_i].token);
                if (!out.empty()) out += ' ';
                out += surface;
            }
            recently_generated.push_back(oscs_[best_i].token);
            if (recently_generated.size() > 32) {
                recently_generated.erase(recently_generated.begin());
            }
            push_active_token(oscs_[best_i].token);
            
            std::string prefix = std::string(text) + " " + out;
            seed_weyl_transform_into(prefix, fp, current_padic, true, true);

            // Quantum Prompt Anchoring (QPA) to keep state trapped in prompt semantic field
            const dz_real alpha = 0.28 / (1.0 + 0.05 * static_cast<dz_real>(s));
            for (std::size_t j = 0; j < dim_; ++j) {
                fp[j] = (1.0 - alpha) * fp[j] + alpha * prompt_trace[j];
            }
            normalize_complex(fp);

            // Gross-Pitaevskii Concept Condensation (GPCC) to attract state to nearest active concepts
            // Two passes: the similarity scan parallelizes over oscillators,
            // then the attraction vector accumulates per dimension in fixed
            // oscillator order — bit-identical to the old serial loop for
            // any worker count.
            std::vector<dz_real> attraction_weight(oscs_.size(), 0.0);
            const auto gpcc_scan = [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    // Structural tokens stay out of concept condensation:
                    // their keys are context anchors, not concepts.
                    if (oscs_[i].token.size() <= 1) continue;
                    if (is_structural_token(oscs_[i].token)) continue;
                    if (is_subword_continuation(oscs_[i].token)) continue;

                    cx dot = 0;
                    for (std::size_t j = 0; j < dim_; ++j) {
                        dot += conjugate_multiply(oscs_[i].key[j], fp[j]);
                    }
                    const dz_real similarity = std::max<dz_real>(0.0, dot.real());
                    if (similarity > 0.04) {
                        attraction_weight[i] = similarity * oscs_[i].strength / (1.0 + oscs_[i].error_ema);
                    }
                }
            };
            if (thread_count_ > 1 && dim_ >= parallel_min_dimensions_ && oscs_.size() >= thread_count_ * 2U) {
                const std::size_t workers = std::min(thread_count_, oscs_.size());
                range_pool(workers).run(oscs_.size(), workers, gpcc_scan);
            } else {
                gpcc_scan(0, oscs_.size());
            }
            std::vector<std::size_t> attracting;
            dz_real total_attr = 0.0;
            for (std::size_t i = 0; i < oscs_.size(); ++i) {
                if (attraction_weight[i] > 0.0) {
                    attracting.push_back(i);
                    total_attr += attraction_weight[i];
                }
            }
            std::vector<cx> concept_attraction(dim_, cx(0, 0));
            if (!attracting.empty()) {
                parallel_for_ranges(dim_, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t j = begin; j < end; ++j) {
                        cx sum = 0;
                        for (const std::size_t i : attracting) {
                            sum += attraction_weight[i] * oscs_[i].key[j];
                        }
                        concept_attraction[j] = sum;
                    }
                });
            }
            if (total_attr > 0.0) {
                const dz_real mu = dimension_interference_ > 0.0 ? 0.015 : 0.16; // damp condensation under interference
                normalize_complex(concept_attraction);
                if (dimension_interference_ > 0.0) {
                    remove_attractor_projection(concept_attraction, attractor_center);
                    remove_attractor_subspace_projection(concept_attraction, attractor_basis);
                }
                for (std::size_t j = 0; j < dim_; ++j) {
                    fp[j] = (1.0 - mu) * fp[j] + mu * concept_attraction[j];
                }
                normalize_complex(fp);
            }

            if (dimension_interference_ > 0.0) {
                apply_conformal_braiding(fp);
            }

            // Dynamic attractor deflection and Cubic NLSE focusing
            if (dimension_interference_ > 0.0) {
                remove_attractor_projection(fp, attractor_center);
                remove_attractor_subspace_projection(fp, attractor_basis);
                remove_attractor_projection(current_padic, attractor_padic_center);
                remove_attractor_subspace_projection(current_padic, attractor_padic_basis);
                
                // Dimensional interference coupling (cross-dimensional mixing)
                apply_dimensional_interference(oscs_[best_i].token, oscs_[best_i].padic_signature, fp);
                
                // Cubic NLSE self-focusing
                const dz_real norm_entropy = calculate_normalized_entropy(fp);
                const dz_real kappa = 0.22 * (1.0 - 0.5 * norm_entropy);
                for (std::size_t j = 0; j < dim_; ++j) {
                    fp[j] += kappa * complex_norm(fp[j]) * fp[j];
                }
                normalize();
            }
        }
        // Rollback working memory (transient Hebbian fast weights)
        for (const auto& saved : saved_oscs) {
            oscs_[saved.index].query = saved.query;
            oscs_[saved.index].transition = saved.transition;
            for (std::size_t p = 0; p < saved.proto_queries.size(); ++p) {
                oscs_[saved.index].prototypes[p].query = saved.proto_queries[p];
                oscs_[saved.index].prototypes[p].transition = saved.proto_transitions[p];
            }
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
        loss_ema_ = 0.0;
    }
    std::size_t observation_count() const noexcept { return total_observations_; }
    std::size_t contrastive_update_count() const noexcept { return contrastive_updates_; }
    dz_real mean_loss() const noexcept { return loss_updates_ == 0 ? 0.0 : loss_ema_; }
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
            const dz_real left_rank = left.strength * std::log1p(static_cast<dz_real>(left.observations));
            const dz_real right_rank = right.strength * std::log1p(static_cast<dz_real>(right.observations));
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
        dz_real total_obs = 0.0;
        for (std::size_t i = 0; i < oscs_.size(); ++i) {
            total_obs += static_cast<dz_real>(oscs_[i].observations);
        }

        for (std::size_t i = 0; i < oscs_.size(); ++i) {
            if (i == found->second) {
                continue;
            }
            const auto& candidate = oscs_[i];
            const dz_real next_similarity = complex_similarity(anchor.query, candidate.key);
            const dz_real context_similarity = complex_similarity(anchor.key, candidate.key);
            const dz_real transition_similarity = complex_similarity(anchor.transition, candidate.transition);
            const dz_real padic_similarity =
                0.5 + 0.5 * cosine(anchor.query_padic_signature, candidate.padic_signature);
            const dz_real reliability = 0.65 + 0.35 * std::clamp<dz_real>(candidate.strength / 4.0, 0.0, 1.0);
            const dz_real idf = std::log(1.0 + total_obs / (1.0 + static_cast<dz_real>(candidate.observations)));
            const dz_real association_score =
                idf * reliability *
                (0.45 * next_similarity +
                 0.25 * context_similarity +
                 0.20 * transition_similarity +
                 0.10 * padic_similarity);
            if (association_score > 1.0e-12) {
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

    void set_generation_temperature(dz_real temperature) noexcept {
        generation_temperature_ = std::clamp<dz_real>(temperature, 0.0, 2.0);
    }

    void set_learning_rate(dz_real rate) noexcept {
        learning_rate_ = std::clamp<dz_real>(rate, 1.0e-4, 1.0);
    }

    void set_update_probability(dz_real probability) noexcept {
        update_probability_ = std::clamp<dz_real>(probability, 0.0, 1.0);
    }

    void set_update_noise(dz_real scale) noexcept {
        update_noise_ = std::clamp<dz_real>(scale, 0.0, 0.25);
    }

    void set_random_init_scale(dz_real scale) noexcept {
        random_init_scale_ = std::clamp<dz_real>(scale, 0.0, 0.25);
    }

    void set_dimension_interference(dz_real strength) noexcept {
        dimension_interference_ = std::clamp<dz_real>(strength, 0.0, 0.25);
    }

    // Conditional-contrast scoring: how strongly the drive discounts a
    // candidate's fit to the global corpus center (half-PMI — a token wins
    // by fitting THIS context better than contexts in general). Applied
    // ramped by dimension interference.
    void set_contrast_beta(dz_real beta) noexcept {
        contrast_beta_ = std::clamp<dz_real>(beta, 0.0, 2.0);
    }

    // Surprise gating: floor of the learning-rate gate for contexts the
    // memory already predicts (1.0 disables gating entirely).
    void set_surprise_floor(dz_real floor) noexcept {
        surprise_floor_ = std::clamp<dz_real>(floor, 0.05, 1.0);
    }

    dz_real dimension_interference() const noexcept { return dimension_interference_; }

    void set_thread_count(std::size_t count) noexcept {
        const auto fallback = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        thread_count_ = count == 0 ? fallback : std::clamp<std::size_t>(count, 1, 1024);
        range_pool_.reset();
    }

    std::size_t thread_count() const noexcept { return thread_count_; }

    void set_parallel_min_dimensions(std::size_t dimensions) noexcept {
        parallel_min_dimensions_ = std::max<std::size_t>(1, dimensions);
    }

    // compact=false writes the exact v2 byte layout. compact=true writes the
    // opt-in v3 format: per-vector max-abs int16 quantization with a float32
    // scale (~8x smaller vectors). load_model auto-detects the version.
    void save_model(std::string_view path, bool compact = false) const {
        std::ofstream output(std::string(path), std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot open model for write: " + std::string(path));
        }
        write_model(output, compact);
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
        std::vector<dz_real> padic_signature;
        std::vector<dz_real> query_padic_signature;
        std::vector<cx> negative_key;
        std::vector<dz_real> negative_padic_signature;
        std::vector<std::uint64_t> context_tail;
        dz_real error_ema = 1.0;
        std::size_t observations = 0;
    };

    struct TokenOscillator {
        std::string token;
        std::vector<cx> query;    // next-state projection (where to go)
        std::vector<cx> key;      // current-state projection (where we are)
        std::vector<cx> transition; // learned phase bridge from key to query
        std::vector<dz_real> padic_signature;
        std::vector<dz_real> query_padic_signature;
        std::vector<cx> negative_key;
        std::vector<dz_real> negative_padic_signature;
        dz_real strength = 1.0;
        dz_real error_ema = 1.0;
        std::size_t observations = 0;
        std::vector<ContextPrototype> prototypes;
    };

    static constexpr std::string_view model_magic() noexcept { return "DZETA_OSC_FIELD"; }
    static constexpr std::uint32_t model_version() noexcept { return 2U; }
    static constexpr std::uint32_t model_version_compact() noexcept { return 3U; }
    // Scheme tag inside v3 files so a future codec does not need another
    // version bump; 0 is reserved as invalid to catch zero-filled corruption.
    static constexpr std::uint8_t quant_scheme_int16_maxabs = 1U;
    static constexpr std::size_t max_serialized_string_bytes = 64U * 1024U * 1024U;
    static constexpr std::size_t max_serialized_vector_items = 16U * 1024U * 1024U;

    template <typename T>
    static void write_pod(std::ostream& output, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if constexpr (std::is_same_v<T, long double> &&
                      std::numeric_limits<long double>::digits == 64) {
            // x86 80-bit long double occupies 10 of sizeof(long double)
            // bytes; the tail is uninitialized padding. Zero it so two saves
            // of the same state produce byte-identical model files and no
            // stack memory leaks into them. The on-disk layout is unchanged.
            unsigned char buffer[sizeof(long double)] = {};
            std::memcpy(buffer, &value, 10U);
            output.write(reinterpret_cast<const char*>(buffer), sizeof(buffer));
        } else {
            output.write(reinterpret_cast<const char*>(&value), sizeof(T));
        }
    }

    // Floating-point payloads cross the serialization boundary as
    // long double regardless of dz_real, so the v2 file format is identical
    // for double and long double builds.
    static void write_real(std::ostream& output, dz_real value) {
        write_pod(output, static_cast<long double>(value));
    }

    static void read_real(std::istream& input, dz_real& value) {
        long double stored = 0.0L;
        read_pod(input, stored);
        value = static_cast<dz_real>(stored);
    }

    static void write_real_vector(std::ostream& output, const std::vector<dz_real>& values) {
        write_count(output, values.size());
        for (const auto& value : values) {
            write_pod(output, static_cast<long double>(value));
        }
    }

    static void read_real_vector(std::istream& input,
                                 std::vector<dz_real>& values,
                                 std::string_view label) {
        const std::size_t size = read_count(input, label);
        if (size > max_serialized_vector_items) {
            throw std::runtime_error("model vector is too large: " + std::string(label));
        }
        values.resize(size);
        for (auto& value : values) {
            long double stored = 0.0L;
            read_pod(input, stored);
            value = static_cast<dz_real>(stored);
        }
        if (!input) {
            throw std::runtime_error("truncated model vector: " + std::string(label));
        }
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
            write_pod(output, static_cast<long double>(value.real()));
            write_pod(output, static_cast<long double>(value.imag()));
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
            value = cx(static_cast<dz_real>(real), static_cast<dz_real>(imag));
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

    // ---- v3 compact quantization -------------------------------------
    // Per-vector max-abs int16 with a float32 scale. All stored vectors are
    // unit-normalized (or near it), so the 15-bit grid perturbs dot products
    // by ~1e-6 — orders of magnitude below inter-candidate score gaps.
    static void write_quantized_complex_vector(std::ostream& output, const std::vector<cx>& values) {
        write_count(output, values.size());
        long double max_abs = 0.0L;
        for (const auto& value : values) {
            const long double re = std::abs(static_cast<long double>(value.real()));
            const long double im = std::abs(static_cast<long double>(value.imag()));
            if (std::isfinite(re)) max_abs = std::max(max_abs, re);
            if (std::isfinite(im)) max_abs = std::max(max_abs, im);
        }
        float scale = max_abs > 0.0L ? static_cast<float>(max_abs / 32767.0L) : 0.0f;
        if (!std::isfinite(scale)) {
            // Absurd magnitudes would store an infinite scale that the
            // reader rejects, leaving a permanently unloadable file; store
            // the vector as zeros instead.
            scale = 0.0f;
        }
        write_pod(output, scale);
        const auto quantize = [&](dz_real raw) -> std::int16_t {
            if (scale <= 0.0f || !std::isfinite(static_cast<long double>(raw))) {
                return 0;
            }
            const long long rounded = std::llround(static_cast<long double>(raw) / static_cast<long double>(scale));
            return static_cast<std::int16_t>(std::clamp<long long>(rounded, -32767, 32767));
        };
        for (const auto& value : values) {
            write_pod(output, quantize(value.real()));
            write_pod(output, quantize(value.imag()));
        }
    }

    static void read_quantized_complex_vector(std::istream& input,
                                              std::vector<cx>& values,
                                              std::string_view label) {
        const std::size_t size = read_count(input, label);
        if (size > max_serialized_vector_items) {
            throw std::runtime_error("model complex vector is too large: " + std::string(label));
        }
        float scale = 0.0f;
        read_pod(input, scale);
        if (!std::isfinite(scale) || scale < 0.0f) {
            throw std::runtime_error("model quantization scale invalid: " + std::string(label));
        }
        values.resize(size);
        for (auto& value : values) {
            std::int16_t real_q = 0;
            std::int16_t imag_q = 0;
            read_pod(input, real_q);
            read_pod(input, imag_q);
            value = cx(static_cast<dz_real>(static_cast<long double>(real_q) * static_cast<long double>(scale)),
                       static_cast<dz_real>(static_cast<long double>(imag_q) * static_cast<long double>(scale)));
        }
        if (!input) {
            throw std::runtime_error("truncated model complex vector: " + std::string(label));
        }
    }

    static void write_quantized_real_vector(std::ostream& output, const std::vector<dz_real>& values) {
        write_count(output, values.size());
        long double max_abs = 0.0L;
        for (const auto value : values) {
            const long double magnitude = std::abs(static_cast<long double>(value));
            if (std::isfinite(magnitude)) max_abs = std::max(max_abs, magnitude);
        }
        float scale = max_abs > 0.0L ? static_cast<float>(max_abs / 32767.0L) : 0.0f;
        if (!std::isfinite(scale)) {
            // Absurd magnitudes would store an infinite scale that the
            // reader rejects, leaving a permanently unloadable file; store
            // the vector as zeros instead.
            scale = 0.0f;
        }
        write_pod(output, scale);
        for (const auto value : values) {
            std::int16_t quantized = 0;
            if (scale > 0.0f && std::isfinite(static_cast<long double>(value))) {
                const long long rounded =
                    std::llround(static_cast<long double>(value) / static_cast<long double>(scale));
                quantized = static_cast<std::int16_t>(std::clamp<long long>(rounded, -32767, 32767));
            }
            write_pod(output, quantized);
        }
    }

    static void read_quantized_real_vector(std::istream& input,
                                           std::vector<dz_real>& values,
                                           std::string_view label) {
        const std::size_t size = read_count(input, label);
        if (size > max_serialized_vector_items) {
            throw std::runtime_error("model vector is too large: " + std::string(label));
        }
        float scale = 0.0f;
        read_pod(input, scale);
        if (!std::isfinite(scale) || scale < 0.0f) {
            throw std::runtime_error("model quantization scale invalid: " + std::string(label));
        }
        values.resize(size);
        for (auto& value : values) {
            std::int16_t quantized = 0;
            read_pod(input, quantized);
            value = static_cast<dz_real>(static_cast<long double>(quantized) * static_cast<long double>(scale));
        }
        if (!input) {
            throw std::runtime_error("truncated model vector: " + std::string(label));
        }
    }

    static void write_context_prototype(std::ostream& output,
                                        const ContextPrototype& prototype,
                                        bool compact) {
        const auto put_complex = [&](const std::vector<cx>& values) {
            compact ? write_quantized_complex_vector(output, values)
                    : write_complex_vector(output, values);
        };
        const auto put_real = [&](const std::vector<dz_real>& values) {
            compact ? write_quantized_real_vector(output, values)
                    : write_real_vector(output, values);
        };
        put_complex(prototype.key);
        put_complex(prototype.query);
        put_complex(prototype.transition);
        put_real(prototype.padic_signature);
        put_real(prototype.query_padic_signature);
        put_complex(prototype.negative_key);
        put_real(prototype.negative_padic_signature);
        write_scalar_vector(output, prototype.context_tail);  // exact uint64 hashes
        write_real(output, prototype.error_ema);
        write_count(output, prototype.observations);
    }

    ContextPrototype read_context_prototype(std::istream& input, bool quantized) const {
        ContextPrototype prototype;
        const auto get_complex = [&](std::vector<cx>& values, std::string_view label) {
            quantized ? read_quantized_complex_vector(input, values, label)
                      : read_complex_vector(input, values, label);
        };
        const auto get_real = [&](std::vector<dz_real>& values, std::string_view label) {
            quantized ? read_quantized_real_vector(input, values, label)
                      : read_real_vector(input, values, label);
        };
        get_complex(prototype.key, "prototype.key");
        get_complex(prototype.query, "prototype.query");
        get_complex(prototype.transition, "prototype.transition");
        get_real(prototype.padic_signature, "prototype.padic_signature");
        get_real(prototype.query_padic_signature, "prototype.query_padic_signature");
        get_complex(prototype.negative_key, "prototype.negative_key");
        get_real(prototype.negative_padic_signature, "prototype.negative_padic_signature");
        read_scalar_vector(input, prototype.context_tail, "prototype.context_tail");
        read_real(input, prototype.error_ema);
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

    static void write_token_oscillator(std::ostream& output,
                                       const TokenOscillator& oscillator,
                                       bool compact) {
        const auto put_complex = [&](const std::vector<cx>& values) {
            compact ? write_quantized_complex_vector(output, values)
                    : write_complex_vector(output, values);
        };
        const auto put_real = [&](const std::vector<dz_real>& values) {
            compact ? write_quantized_real_vector(output, values)
                    : write_real_vector(output, values);
        };
        write_string(output, oscillator.token);
        put_complex(oscillator.query);
        put_complex(oscillator.key);
        put_complex(oscillator.transition);
        put_real(oscillator.padic_signature);
        put_real(oscillator.query_padic_signature);
        put_complex(oscillator.negative_key);
        put_real(oscillator.negative_padic_signature);
        write_real(output, oscillator.strength);
        write_real(output, oscillator.error_ema);
        write_count(output, oscillator.observations);
        write_count(output, oscillator.prototypes.size());
        for (const auto& prototype : oscillator.prototypes) {
            write_context_prototype(output, prototype, compact);
        }
    }

    TokenOscillator read_token_oscillator(std::istream& input, bool quantized) const {
        TokenOscillator oscillator;
        const auto get_complex = [&](std::vector<cx>& values, std::string_view label) {
            quantized ? read_quantized_complex_vector(input, values, label)
                      : read_complex_vector(input, values, label);
        };
        const auto get_real = [&](std::vector<dz_real>& values, std::string_view label) {
            quantized ? read_quantized_real_vector(input, values, label)
                      : read_real_vector(input, values, label);
        };
        read_string(input, oscillator.token, "oscillator.token");
        get_complex(oscillator.query, "oscillator.query");
        get_complex(oscillator.key, "oscillator.key");
        get_complex(oscillator.transition, "oscillator.transition");
        get_real(oscillator.padic_signature, "oscillator.padic_signature");
        get_real(oscillator.query_padic_signature, "oscillator.query_padic_signature");
        get_complex(oscillator.negative_key, "oscillator.negative_key");
        get_real(oscillator.negative_padic_signature, "oscillator.negative_padic_signature");
        read_real(input, oscillator.strength);
        read_real(input, oscillator.error_ema);
        oscillator.observations = read_count(input, "oscillator.observations");
        const std::size_t prototype_count = read_count(input, "oscillator.prototypes");
        if (prototype_count > 1024U) {
            throw std::runtime_error("model has too many prototypes for token: " + oscillator.token);
        }
        oscillator.prototypes.reserve(prototype_count);
        for (std::size_t i = 0; i < prototype_count; ++i) {
            oscillator.prototypes.push_back(read_context_prototype(input, quantized));
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

    void write_model(std::ostream& output, bool compact = false) const {
        write_string(output, model_magic());
        write_pod(output, compact ? model_version_compact() : model_version());
        if (compact) {
            write_pod(output, quant_scheme_int16_maxabs);
        }
        write_count(output, max_osc_);
        write_count(output, dim_);
        write_count(output, thread_count_);
        write_count(output, parallel_min_dimensions_);
        write_real(output, learning_rate_);
        write_real(output, generation_temperature_);
        write_real(output, contrastive_rate_);
        write_real(output, contrastive_margin_);
        write_real(output, contrastive_strength_);
        write_real(output, update_probability_);
        write_real(output, update_noise_);
        write_real(output, random_init_scale_);
        write_real(output, dimension_interference_);
        write_count(output, max_prototypes_per_token_);
        write_count(output, max_hard_negatives_);
        write_count(output, max_context_tokens_);
        write_count(output, contrastive_period_);
        write_count(output, total_observations_);
        write_count(output, contrastive_updates_);
        write_count(output, loss_updates_);
        write_real(output, loss_ema_);

        std::ostringstream rng_state;
        rng_state << rng_;
        write_string(output, rng_state.str());

        write_count(output, oscs_.size());
        for (const auto& oscillator : oscs_) {
            write_token_oscillator(output, oscillator, compact);
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
        if (version == 0 || version > model_version_compact()) {
            throw std::runtime_error("unsupported dzeta oscillator model version");
        }
        const bool quantized = version >= model_version_compact();
        if (quantized) {
            std::uint8_t scheme = 0;
            read_pod(input, scheme);
            if (!input || scheme != quant_scheme_int16_maxabs) {
                throw std::runtime_error("unsupported dzeta model quantization scheme");
            }
        }

        max_osc_ = std::max<std::size_t>(128, read_count(input, "max_osc"));
        dim_ = std::max<std::size_t>(16, read_count(input, "dimensions"));
        thread_count_ = std::max<std::size_t>(1, read_count(input, "thread_count"));
        range_pool_.reset();
        parallel_min_dimensions_ = std::max<std::size_t>(1, read_count(input, "parallel_min_dimensions"));
        read_real(input, learning_rate_);
        read_real(input, generation_temperature_);
        read_real(input, contrastive_rate_);
        read_real(input, contrastive_margin_);
        read_real(input, contrastive_strength_);
        read_real(input, update_probability_);
        read_real(input, update_noise_);
        read_real(input, random_init_scale_);
        if (version >= 2U) {
            read_real(input, dimension_interference_);
        } else {
            dimension_interference_ = 0.0;
        }
        max_prototypes_per_token_ = std::max<std::size_t>(1, read_count(input, "max_prototypes_per_token"));
        max_hard_negatives_ = std::max<std::size_t>(1, read_count(input, "max_hard_negatives"));
        max_context_tokens_ = std::max<std::size_t>(1, read_count(input, "max_context_tokens"));
        contrastive_period_ = std::max<std::size_t>(1, read_count(input, "contrastive_period"));
        total_observations_ = read_count(input, "total_observations");
        contrastive_updates_ = read_count(input, "contrastive_updates");
        loss_updates_ = read_count(input, "loss_updates");
        read_real(input, loss_ema_);

        std::string rng_state;
        read_string(input, rng_state, "rng_state");
        std::istringstream rng_input(rng_state);
        rng_input >> rng_;
        if (!rng_input) {
            rng_.seed(entropy_seed());
        }

        initialize_spectral_basis();
        initialize_seed_projection(64);

        const std::size_t oscillator_count = read_count(input, "oscillators");
        if (oscillator_count > max_osc_) {
            max_osc_ = oscillator_count;
        }
        oscs_.clear();
        token_index_.clear();
        oscs_.reserve(oscillator_count);
        for (std::size_t i = 0; i < oscillator_count; ++i) {
            auto oscillator = read_token_oscillator(input, quantized);
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
                                   std::vector<dz_real>(dim_, 0.0),
                                   std::vector<dz_real>(dim_, 0.0),
                                   std::vector<cx>(dim_, cx(0, 0)),
                                   std::vector<dz_real>(dim_, 0.0),
                                   1.0,
                                   1.0,
                                   0,
                                   {}};
        if (random_init_scale_ > 0.0) {
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
        // seed_weyl_transform_from_signature keeps per-prime dither state in
        // 128-entry stack arrays; the seed prime count must never exceed that.
        count = std::min<std::size_t>(count, 128);
        seed_primes_ = generate_first_primes(std::max<std::size_t>(1, count));
        seed_theta_.resize(seed_primes_.size());
        seed_energy_.resize(seed_primes_.size());
        seed_padic_log_.resize(seed_primes_.size());
        seed_prime_phase_.resize(seed_primes_.size());
        for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
            seed_theta_[i] = riemann_siegel_theta(seed_primes_[i]);
            seed_energy_[i] = spectral_energy(seed_primes_[i], 32);
            seed_padic_log_[i] = std::log1p(static_cast<dz_real>(seed_primes_[i] % 997U));
            seed_prime_phase_[i] = static_cast<dz_real>(seed_primes_[i]) * 0.0174533;
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
        // Grow-only: run() clamps to the requested worker count and surplus
        // pool threads idle, so smaller jobs reuse a larger pool. The old
        // exact-match condition destroyed and respawned every OS thread each
        // time consecutive sections requested different counts — up to twice
        // per generated token.
        if (!range_pool_ || range_pool_->max_workers() < workers) {
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

    static dz_real complex_norm(cx value) noexcept {
        const dz_real re = value.real();
        const dz_real im = value.imag();
        return re * re + im * im;
    }

    static cx conjugate_multiply(cx left, cx right) noexcept {
        const dz_real lr = left.real();
        const dz_real li = left.imag();
        const dz_real rr = right.real();
        const dz_real ri = right.imag();
        return {lr * rr + li * ri, lr * ri - li * rr};
    }

    static dz_real calculate_normalized_entropy(const std::vector<cx>& state) {
        dz_real entropy = 0.0;
        for (const auto& v : state) {
            const dz_real p = complex_norm(v);
            if (p > 1.0e-15) {
                entropy -= p * std::log(p);
            }
        }
        const dz_real max_entropy = std::log(static_cast<dz_real>(state.size()));
        return max_entropy > 0.0 ? std::clamp<dz_real>(entropy / max_entropy, 0.0, 1.0) : 0.0;
    }

    static void sincos_ld(dz_real value, dz_real& sin_value, dz_real& cos_value) noexcept {
        sin_value = std::sin(value);
        cos_value = std::cos(value);
    }

    void seed_weyl_transform_from_signature(const std::vector<dz_real>& signature,
                                            std::vector<cx>& ampl,
                                            std::vector<dz_real>& padic,
                                            std::string_view impulse_for_di,
                                            bool enable_dimensional_interference = true,
                                            bool allow_parallel = true) const {
        ampl.resize(dim_);
        padic.assign(dim_, 0.0);
        if (seed_primes_.empty()) {
            return;
        }

        std::vector<dz_real> seed_weight(signature.size(), 0.0);
        dz_real padic_base = 0.0;
        for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
            const dz_real charge = signature[i];
            const dz_real act = std::clamp<dz_real>(0.45 + 0.35 * std::abs(charge), 0.0, 1.0);
            const dz_real en = seed_energy_[i];
            seed_weight[i] = act * en;
            const dz_real padic_coordinate = seed_padic_log_[i] * (1.0 + charge);
            padic_base += act * en * padic_norm(padic_coordinate, seed_primes_[i % seed_primes_.size()] % 997U + 2U);
        }
        if (std::abs(padic_base) > 1.0e-30) {
            std::fill(padic.begin(), padic.end(), padic_base);
        }

        const auto compute_range = [&](std::size_t begin, std::size_t end) {
            constexpr std::size_t seed_stack_capacity = 128;
            // The dither recurrence restarts from the closed form at fixed
            // block boundaries (not at the range begin), so any partition of
            // [0, dim) over any worker count produces bit-identical output.
            // The old per-range restart made learned state depend on the
            // thread count at the last-ulp level.
            constexpr std::size_t dither_block = 64;
            std::array<dz_real, seed_stack_capacity> dither_sin{};
            std::array<dz_real, seed_stack_capacity> dither_cos{};
            std::array<dz_real, seed_stack_capacity> step_sin{};
            std::array<dz_real, seed_stack_capacity> step_cos{};
            const auto init_dither_at = [&](std::size_t z0) {
                for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
                    const dz_real charge = signature[i];
                    const dz_real start_phase =
                        charge * (static_cast<dz_real>(z0) + 1.0) * 12.9898 + seed_prime_phase_[i];
                    sincos_ld(start_phase, dither_sin[i], dither_cos[i]);
                }
            };
            const auto advance_dither = [&]() {
                for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
                    const dz_real next_sin =
                        dither_sin[i] * step_cos[i] + dither_cos[i] * step_sin[i];
                    const dz_real next_cos =
                        dither_cos[i] * step_cos[i] - dither_sin[i] * step_sin[i];
                    dither_sin[i] = next_sin;
                    dither_cos[i] = next_cos;
                }
            };
            for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
                const dz_real step_phase = signature[i] * 12.9898;
                sincos_ld(step_phase, step_sin[i], step_cos[i]);
            }
            const std::size_t block_start = begin - (begin % dither_block);
            init_dither_at(block_start);
            for (std::size_t z = block_start; z < begin; ++z) {
                advance_dither();
            }
            for (std::size_t z = begin; z < end; ++z) {
                if (z != block_start && z % dither_block == 0) {
                    init_dither_at(z);
                }
                dz_real sum_re = 0.0;
                dz_real sum_im = 0.0;
                const dz_real zr = zeta_basis_[z];
                for (std::size_t i = 0; i < seed_primes_.size(); ++i) {
                    const dz_real charge = signature[i];
                    const dz_real theta = seed_theta_[i];
                    const dz_real phase_dither = 0.03 * dither_sin[i];
                    const dz_real phase = theta * zr + charge * 0.5 + phase_dither;
                    const dz_real weight = seed_weight[i];
                    dz_real sin_phase = 0.0;
                    dz_real cos_phase = 1.0;
                    sincos_ld(phase, sin_phase, cos_phase);
                    sum_re += weight * cos_phase;
                    sum_im += weight * sin_phase;
                }
                advance_dither();
                ampl[z] = cx(sum_re, sum_im);
            }
        };
        if (allow_parallel) {
            parallel_for_ranges(dim_, compute_range);
        } else {
            compute_range(0, dim_);
        }

        dz_real an = 0.0;
        for (auto value : ampl) an += complex_norm(value);
        if (an > 1.0e-30) {
            an = std::sqrt(an);
            for (auto& value : ampl) value /= an;
        }
        if (enable_dimensional_interference && !impulse_for_di.empty()) {
            apply_dimensional_interference(impulse_for_di, signature, ampl);
        }
        dz_real pn = 0.0;
        for (auto value : padic) pn += value * value;
        if (pn > 1.0e-30) {
            pn = std::sqrt(pn);
            for (auto& value : padic) value /= pn;
        }
    }

    void seed_weyl_transform_into(std::string_view impulse,
                                  std::vector<cx>& ampl,
                                  std::vector<dz_real>& padic,
                                  bool enable_dimensional_interference = true,
                                  bool allow_parallel = true) const {
        const auto raw_signature = field_impulse_signature(impulse, seed_primes_.size());
        const std::vector<dz_real> signature(raw_signature.begin(), raw_signature.end());
        seed_weyl_transform_from_signature(signature, ampl, padic, impulse, enable_dimensional_interference, allow_parallel);
    }

    std::pair<std::vector<cx>, std::vector<dz_real>> seed_weyl_transform(std::string_view impulse,
                                                                             bool allow_parallel = true) const {
        std::vector<cx> ampl;
        std::vector<dz_real> padic;
        seed_weyl_transform_into(impulse, ampl, padic, true, allow_parallel);
        return {std::move(ampl), std::move(padic)};
    }

    void apply_dimensional_interference(std::string_view impulse,
                                        const std::vector<dz_real>& signature,
                                        std::vector<cx>& ampl) const {
        if (dimension_interference_ <= 0.0 || ampl.size() < 8 || signature.empty()) {
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

        const dz_real dim_gain =
            std::sqrt(std::log2(static_cast<dz_real>(count) + 2.0) / std::log2(194.0));
        const dz_real strength = std::min<dz_real>(0.65, dimension_interference_ * dim_gain);
        const dz_real fold_scale = std::sqrt(static_cast<dz_real>(count));

        for (std::size_t z = 0; z < count; ++z) {
            const std::size_t ia = (z + shift_a) % count;
            const std::size_t ib = (z * stride + shift_b) % count;
            const std::size_t ic = (z + shift_c) % count;
            const dz_real charge = signature[z % signature.size()];
            const dz_real gate = 0.55 + 0.45 * std::abs(charge);
            const cx phase_gate(1.0 + 0.15 * charge, 0.35 * charge);
            const cx folded =
                fold_scale * (source[ia] * std::conj(source[ib]) + dz_real(0.5) * source[z] * std::conj(source[ic]));
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

    void build_prompt_anchor_field(std::string_view text,
                                   const std::vector<cx>& attractor_center,
                                   const std::vector<dz_real>& attractor_padic_center,
                                   const std::vector<std::vector<cx>>& attractor_basis,
                                   const std::vector<std::vector<dz_real>>& attractor_padic_basis,
                                   std::vector<cx>& anchor_field,
                                   std::vector<dz_real>& anchor_padic) const {
        anchor_field.assign(dim_, cx(0, 0));
        anchor_padic.assign(dim_, 0.0);
        const auto tokens = tokenize_query(text);
        if (tokens.empty()) {
            anchor_field.clear();
            anchor_padic.clear();
            return;
        }

        dz_real total_weight = 0.0;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].size() <= 1) {
                continue;
            }
            const auto* oscillator = find_prompt_oscillator(tokens[i]);
            if (oscillator == nullptr) {
                continue;
            }
            const dz_real recency =
                1.0 + static_cast<dz_real>(i) / std::max<dz_real>(1.0, tokens.size());
            const dz_real weight =
                recency * std::clamp<dz_real>(static_cast<dz_real>(tokens[i].size()) / 6.0, 0.55, 1.75) *
                std::clamp<dz_real>(std::log1p(static_cast<dz_real>(oscillator->observations)) + 0.7,
                           0.7,
                           3.0);
            for (std::size_t j = 0; j < dim_; ++j) {
                anchor_field[j] += weight * (dz_real(0.26) * oscillator->key[j] +
                                             dz_real(0.62) * oscillator->query[j] +
                                             dz_real(0.22) * oscillator->transition[j] -
                                             dz_real(0.42) * oscillator->negative_key[j]);
                anchor_padic[j] +=
                    weight * (0.32 * oscillator->padic_signature[j] +
                              0.68 * oscillator->query_padic_signature[j] -
                              0.38 * oscillator->negative_padic_signature[j]);
            }
            total_weight += weight;
        }

        if (total_weight <= 1.0e-18) {
            anchor_field.clear();
            anchor_padic.clear();
            return;
        }
        const dz_real inv_weight = 1.0 / total_weight;
        for (auto& value : anchor_field) {
            value *= inv_weight;
        }
        for (auto& value : anchor_padic) {
            value *= inv_weight;
        }
        normalize_complex(anchor_field);
        normalize_real(anchor_padic);
        remove_attractor_projection(anchor_field, attractor_center);
        remove_attractor_projection(anchor_padic, attractor_padic_center);
        remove_attractor_subspace_projection(anchor_field, attractor_basis);
        remove_attractor_subspace_projection(anchor_padic, attractor_padic_basis);
    }

    void build_attractor_center(std::vector<cx>& center, std::vector<dz_real>& padic_center) const {
        center.assign(dim_, cx(0, 0));
        padic_center.assign(dim_, 0.0);
        dz_real total_weight = 0.0;
        for (const auto& oscillator : oscs_) {
            if (oscillator.observations == 0 || oscillator.token.size() <= 1 ||
                is_structural_token(oscillator.token) ||
                is_subword_continuation(oscillator.token)) {
                continue;
            }
            const dz_real observations = static_cast<dz_real>(oscillator.observations);
            const dz_real weight =
                std::sqrt(observations) * std::clamp<dz_real>(oscillator.strength / 4.0, 0.10, 2.00);
            if (weight <= 1.0e-18) {
                continue;
            }
            for (std::size_t j = 0; j < dim_; ++j) {
                center[j] += weight * (dz_real(0.55) * oscillator.key[j] + dz_real(0.45) * oscillator.query[j]);
                padic_center[j] +=
                    weight * (0.55 * oscillator.padic_signature[j] + 0.45 * oscillator.query_padic_signature[j]);
            }
            total_weight += weight;
        }
        if (total_weight <= 1.0e-18) {
            center.clear();
            padic_center.clear();
            return;
        }
        const dz_real inv_weight = 1.0 / total_weight;
        for (auto& value : center) {
            value *= inv_weight;
        }
        for (auto& value : padic_center) {
            value *= inv_weight;
        }
        normalize_complex(center);
        normalize_real(padic_center);
    }

    void build_attractor_subspace(const std::vector<cx>& center,
                                  const std::vector<dz_real>& padic_center,
                                  std::vector<std::vector<cx>>& basis,
                                  std::vector<std::vector<dz_real>>& padic_basis) const {
        basis.clear();
        padic_basis.clear();
        if (oscs_.empty() || dim_ == 0) {
            return;
        }

        struct AttractorCandidate {
            dz_real weight;
            std::size_t oscillator;
        };
        std::vector<AttractorCandidate> candidates;
        candidates.reserve(oscs_.size());
        for (std::size_t i = 0; i < oscs_.size(); ++i) {
            const auto& oscillator = oscs_[i];
            if (oscillator.observations == 0 || oscillator.token.size() <= 1 ||
                is_structural_token(oscillator.token) ||
                is_subword_continuation(oscillator.token)) {
                continue;
            }
            const dz_real observations = static_cast<dz_real>(oscillator.observations);
            const dz_real weight =
                std::sqrt(observations) * std::clamp<dz_real>(oscillator.strength / (1.0 + oscillator.error_ema),
                                                     0.05,
                                                     10.0);
            candidates.push_back({weight, i});
        }
        if (candidates.empty()) {
            return;
        }

        const auto by_weight = [](const AttractorCandidate& left, const AttractorCandidate& right) {
            if (left.weight == right.weight) {
                return left.oscillator < right.oscillator;
            }
            return left.weight > right.weight;
        };
        const std::size_t inspect = std::min<std::size_t>(64, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + inspect, candidates.end(), by_weight);
        constexpr std::size_t max_axes = 6;

        for (std::size_t ci = 0; ci < inspect && basis.size() < max_axes; ++ci) {
            const auto& oscillator = oscs_[candidates[ci].oscillator];
            std::vector<cx> direction(dim_, cx(0, 0));
            std::vector<dz_real> padic_direction(dim_, 0.0);
            for (std::size_t j = 0; j < dim_; ++j) {
                direction[j] = dz_real(0.55) * oscillator.key[j] + dz_real(0.45) * oscillator.query[j];
                padic_direction[j] =
                    0.55 * oscillator.padic_signature[j] + 0.45 * oscillator.query_padic_signature[j];
            }

            subtract_complex_projection(direction, center);
            subtract_real_projection(padic_direction, padic_center);
            for (const auto& axis : basis) {
                subtract_complex_projection(direction, axis);
            }
            for (const auto& axis : padic_basis) {
                subtract_real_projection(padic_direction, axis);
            }

            const dz_real field_norm = complex_vector_norm(direction);
            const dz_real padic_norm_value = real_vector_norm(padic_direction);
            if (field_norm <= 1.0e-12 || padic_norm_value <= 1.0e-12) {
                continue;
            }
            for (auto& value : direction) {
                value /= field_norm;
            }
            for (auto& value : padic_direction) {
                value /= padic_norm_value;
            }
            basis.push_back(std::move(direction));
            padic_basis.push_back(std::move(padic_direction));
        }
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
            values[i] -= dz_real(0.02) * projection * center[i];
        }
        normalize_complex(values);
    }

    static void remove_attractor_projection(std::vector<dz_real>& values,
                                            const std::vector<dz_real>& center) {
        const std::size_t count = std::min(values.size(), center.size());
        if (count == 0) {
            return;
        }
        dz_real projection = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            projection += values[i] * center[i];
        }
        for (std::size_t i = 0; i < count; ++i) {
            values[i] -= dz_real(0.02) * projection * center[i];
        }
        normalize_real(values);
    }

    static void remove_attractor_subspace_projection(std::vector<cx>& values,
                                                     const std::vector<std::vector<cx>>& basis) {
        for (const auto& axis : basis) {
            const std::size_t count = std::min(values.size(), axis.size());
            if (count == 0) continue;
            cx projection = 0;
            for (std::size_t i = 0; i < count; ++i) {
                projection += conjugate_multiply(axis[i], values[i]);
            }
            for (std::size_t i = 0; i < count; ++i) {
                values[i] -= dz_real(0.02) * projection * axis[i];
            }
        }
        normalize_complex(values);
    }

    static void remove_attractor_subspace_projection(std::vector<dz_real>& values,
                                                     const std::vector<std::vector<dz_real>>& basis) {
        for (const auto& axis : basis) {
            const std::size_t count = std::min(values.size(), axis.size());
            if (count == 0) continue;
            dz_real projection = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                projection += values[i] * axis[i];
            }
            for (std::size_t i = 0; i < count; ++i) {
                values[i] -= dz_real(0.02) * projection * axis[i];
            }
        }
        normalize_real(values);
    }

    static dz_real complex_subspace_pressure(const std::vector<cx>& values,
                                                 const std::vector<std::vector<cx>>& basis) {
        dz_real energy = 0.0;
        for (const auto& axis : basis) {
            const std::size_t count = std::min(values.size(), axis.size());
            cx projection = 0;
            for (std::size_t i = 0; i < count; ++i) {
                projection += conjugate_multiply(axis[i], values[i]);
            }
            energy += complex_norm(projection);
        }
        return std::clamp<dz_real>(std::sqrt(energy), 0.0, 1.0);
    }

    static dz_real real_subspace_pressure(const std::vector<dz_real>& values,
                                              const std::vector<std::vector<dz_real>>& basis) {
        dz_real energy = 0.0;
        for (const auto& axis : basis) {
            const std::size_t count = std::min(values.size(), axis.size());
            dz_real projection = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                projection += values[i] * axis[i];
            }
            energy += projection * projection;
        }
        return std::clamp<dz_real>(std::sqrt(energy), 0.0, 1.0);
    }

    void apply_conformal_braiding(std::vector<cx>& state) const {
        if (dimension_interference_ <= 0.0) return;
        const dz_real lambda = 0.08; // braiding strength
        std::vector<cx> braided = state;
        parallel_for_ranges(dim_, [&](std::size_t begin, std::size_t end) {
            for (std::size_t j = begin; j < end; ++j) {
                const std::size_t k_plus = (2 * j + 1) % dim_;
                const std::size_t k_minus = (3 * j + 2) % dim_;
                const dz_real phase_plus = std::atan2(state[k_plus].imag(), state[k_plus].real());
                const dz_real phase_minus = std::atan2(state[k_minus].imag(), state[k_minus].real());
                const dz_real delta_phase = phase_plus - phase_minus;

                dz_real st = 0.0, ct = 1.0;
                sincos_ld(lambda * delta_phase, st, ct);
                braided[j] *= cx(ct, st);
            }
        });
        state = std::move(braided);
        normalize_complex(state);
    }

    void apply_prompt_hamiltonian_transport(std::vector<cx>& state,
                                            std::vector<dz_real>& padic,
                                            const std::vector<cx>& prompt_delta,
                                            const std::vector<dz_real>& prompt_padic_delta,
                                            std::size_t step,
                                            dz_real polarity = 1.0) const {
        const std::size_t count = std::min(state.size(), prompt_delta.size());
        if (count == 0) {
            return;
        }
        const dz_real decay = 1.0 / (1.0 + 0.06 * static_cast<dz_real>(step));
        const dz_real phase_strength =
            std::min<dz_real>(0.62, 0.12 + 1.70 * dimension_interference_) * decay;
        const dz_real mix_strength =
            std::min<dz_real>(0.24, 0.04 + 0.58 * dimension_interference_) * decay;
        parallel_for_ranges(count, [&](std::size_t begin, std::size_t end) {
            for (std::size_t j = begin; j < end; ++j) {
                const cx prompt_direction = polarity * prompt_delta[j];
                const cx coupling = conjugate_multiply(prompt_direction, state[j]);
                const dz_real padic_charge =
                    j < padic.size() && j < prompt_padic_delta.size()
                        ? polarity * padic[j] * prompt_padic_delta[j]
                        : 0.0;
                const dz_real phase =
                    phase_strength * std::atan2(coupling.imag() + 0.35 * padic_charge,
                                                1.0 + std::abs(coupling.real()));
                dz_real st = 0.0;
                dz_real ct = 1.0;
                sincos_ld(phase, st, ct);
                const cx transported = state[j] * cx(ct, st);
                const dz_real gate = std::clamp<dz_real>(0.55 + 0.45 * std::abs(padic_charge), 0.20, 1.25);
                state[j] = (1.0 - mix_strength) * transported + mix_strength * gate * prompt_direction;
            }
        });
        const std::size_t padic_count = std::min(padic.size(), prompt_padic_delta.size());
        for (std::size_t j = 0; j < padic_count; ++j) {
            padic[j] = (1.0 - mix_strength) * padic[j] + mix_strength * polarity * prompt_padic_delta[j];
        }
        normalize_complex(state);
        normalize_real(padic);
    }

    void inject_prompt_resonance(std::string_view text,
                                 std::vector<cx>& state,
                                 std::vector<dz_real>& padic) const {
        if (dimension_interference_ <= 0.0 || state.size() != dim_) {
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

        const dz_real base_weight =
            std::min<dz_real>(0.58, dimension_interference_ * 1.85) /
            std::sqrt(static_cast<dz_real>(anchors.size()));
        for (std::size_t a = 0; a < anchors.size(); ++a) {
            const auto& oscillator = *anchors[a];
            const dz_real recency =
                1.0 + static_cast<dz_real>(a) / static_cast<dz_real>(anchors.size());
            const dz_real weight = std::min<dz_real>(0.62, base_weight * recency);
            for (std::size_t j = 0; j < dim_; ++j) {
                const cx anchor = dz_real(0.35) * oscillator.key[j] + dz_real(0.65) * oscillator.query[j];
                state[j] = (1.0 - weight) * state[j] + weight * anchor;
                padic[j] = (1.0 - weight) * padic[j] + weight * oscillator.query_padic_signature[j];
            }
            normalize_complex(state);
            normalize_real(padic);
        }
    }

    static dz_real complex_vector_norm(const std::vector<cx>& values) {
        dz_real norm = 0.0;
        for (auto value : values) {
            norm += complex_norm(value);
        }
        return norm <= 1.0e-30 ? 0.0 : std::sqrt(norm);
    }

    static dz_real real_vector_norm(const std::vector<dz_real>& values) {
        dz_real norm = 0.0;
        for (auto value : values) {
            norm += value * value;
        }
        return norm <= 1.0e-30 ? 0.0 : std::sqrt(norm);
    }

    static void subtract_complex_projection(std::vector<cx>& values, const std::vector<cx>& axis) {
        const std::size_t count = std::min(values.size(), axis.size());
        if (count == 0) {
            return;
        }
        cx projection = 0;
        for (std::size_t i = 0; i < count; ++i) {
            projection += conjugate_multiply(axis[i], values[i]);
        }
        for (std::size_t i = 0; i < count; ++i) {
            values[i] -= projection * axis[i];
        }
    }

    static void subtract_real_projection(std::vector<dz_real>& values,
                                         const std::vector<dz_real>& axis) {
        const std::size_t count = std::min(values.size(), axis.size());
        if (count == 0) {
            return;
        }
        dz_real projection = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            projection += values[i] * axis[i];
        }
        for (std::size_t i = 0; i < count; ++i) {
            values[i] -= projection * axis[i];
        }
    }

    static void normalize_complex(std::vector<cx>& values) {
        dz_real norm = 0.0;
        for (auto value : values) {
            norm += complex_norm(value);
        }
        if (norm <= 1.0e-30) {
            return;
        }
        norm = std::sqrt(norm);
        for (auto& value : values) {
            value /= norm;
        }
    }

    static dz_real complex_loss(const std::vector<cx>& left, const std::vector<cx>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0;
        }
        dz_real loss = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            loss += complex_norm(left[i] - right[i]);
        }
        return loss / static_cast<dz_real>(count);
    }

    static dz_real complex_similarity(const std::vector<cx>& left, const std::vector<cx>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0;
        }
        cx dot = 0;
        dz_real left_norm = 0.0;
        dz_real right_norm = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            dot += conjugate_multiply(left[i], right[i]);
            left_norm += complex_norm(left[i]);
            right_norm += complex_norm(right[i]);
        }
        if (left_norm <= 1.0e-30 || right_norm <= 1.0e-30) {
            return 0.0;
        }
        return std::clamp<dz_real>(std::abs(dot) / std::sqrt(left_norm * right_norm), 0.0, 1.0);
    }

    static dz_real normalized_complex_similarity(const std::vector<cx>& left, const std::vector<cx>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0;
        }
        cx dot = 0;
        for (std::size_t i = 0; i < count; ++i) {
            dot += conjugate_multiply(left[i], right[i]);
        }
        return std::clamp<dz_real>(std::abs(dot), 0.0, 1.0);
    }

    static void spectral_bridge_into(const std::vector<cx>& from,
                                     const std::vector<cx>& to,
                                     std::vector<cx>& bridge) {
        const std::size_t count = std::min(from.size(), to.size());
        bridge.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            const dz_real magnitude = std::max<dz_real>(1.0e-12, std::abs(from[i]));
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

    static dz_real projected_bridge_similarity(const std::vector<cx>& state,
                                                   const std::vector<cx>& bridge,
                                                   const std::vector<cx>& target) {
        const std::size_t count = std::min({state.size(), bridge.size(), target.size()});
        if (count == 0) {
            return 0.0;
        }
        cx dot = 0;
        dz_real projected_norm = 0.0;
        dz_real target_norm = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const cx projected = state[i] * bridge[i];
            dot += conjugate_multiply(projected, target[i]);
            projected_norm += complex_norm(projected);
            target_norm += complex_norm(target[i]);
        }
        if (projected_norm <= 1.0e-30 || target_norm <= 1.0e-30) {
            return 0.0;
        }
        return std::clamp<dz_real>(std::abs(dot) / std::sqrt(projected_norm * target_norm), 0.0, 1.0);
    }

    static void mix_negative_key(std::vector<cx>& negative_key,
                                 const std::vector<cx>& target_key,
                                 dz_real rate) {
        const std::size_t count = std::min(negative_key.size(), target_key.size());
        for (std::size_t i = 0; i < count; ++i) {
            negative_key[i] = (1.0 - rate) * negative_key[i] + rate * target_key[i];
        }
        normalize_complex(negative_key);
    }

    static void mix_negative_padic(std::vector<dz_real>& negative_padic,
                                   const std::vector<dz_real>& target_padic,
                                   dz_real rate) {
        const std::size_t count = std::min(negative_padic.size(), target_padic.size());
        for (std::size_t i = 0; i < count; ++i) {
            negative_padic[i] = (1.0 - rate) * negative_padic[i] + rate * target_padic[i];
        }
        normalize_real(negative_padic);
    }

    static void repel_from(std::vector<cx>& vector,
                           const std::vector<cx>& away,
                           dz_real rate) {
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

    // cosine() with both norms supplied by the caller. The candidate-side
    // norms are step-invariant and the state-side norm is shared by every
    // candidate in a generation step, so recomputing them per call tripled
    // the p-adic scoring cost. Returns exactly what cosine() would.
    static dz_real cosine_cached(const std::vector<dz_real>& left,
                                     const std::vector<dz_real>& right,
                                     dz_real left_norm,
                                     dz_real right_norm) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0;
        }
        if (left_norm <= 1.0e-30 || right_norm <= 1.0e-30) {
            return 0.0;
        }
        dz_real dot = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            dot += left[i] * right[i];
        }
        return std::clamp<dz_real>(dot / std::sqrt(left_norm * right_norm), -1.0, 1.0);
    }

    // complex_similarity() with cached norms; skips the O(dim) dot product
    // entirely when either side is all-zero (fresh negative memories).
    static dz_real complex_similarity_cached(const std::vector<cx>& left,
                                                 const std::vector<cx>& right,
                                                 dz_real left_norm,
                                                 dz_real right_norm) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0;
        }
        if (left_norm <= 1.0e-30 || right_norm <= 1.0e-30) {
            return 0.0;
        }
        cx dot = 0;
        for (std::size_t i = 0; i < count; ++i) {
            dot += conjugate_multiply(left[i], right[i]);
        }
        return std::clamp<dz_real>(std::abs(dot) / std::sqrt(left_norm * right_norm), 0.0, 1.0);
    }

    static dz_real cosine(const std::vector<dz_real>& left, const std::vector<dz_real>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0;
        }
        dz_real dot = 0.0;
        dz_real left_norm = 0.0;
        dz_real right_norm = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            dot += left[i] * right[i];
            left_norm += left[i] * left[i];
            right_norm += right[i] * right[i];
        }
        if (left_norm <= 1.0e-30 || right_norm <= 1.0e-30) {
            return 0.0;
        }
        return std::clamp<dz_real>(dot / std::sqrt(left_norm * right_norm), -1.0, 1.0);
    }

    static dz_real normalized_cosine(const std::vector<dz_real>& left,
                                         const std::vector<dz_real>& right) {
        const std::size_t count = std::min(left.size(), right.size());
        if (count == 0) {
            return 0.0;
        }
        dz_real dot = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            dot += left[i] * right[i];
        }
        return std::clamp<dz_real>(dot, -1.0, 1.0);
    }

    static void normalize_real(std::vector<dz_real>& values) {
        dz_real norm = 0.0;
        for (auto value : values) {
            norm += value * value;
        }
        if (norm <= 1.0e-30) {
            return;
        }
        norm = std::sqrt(norm);
        for (auto& value : values) {
            value /= norm;
        }
    }

    // Lowercased alnum runs of a raw code token — the same normalization
    // tokenize_query applies to raw text, so learn-time streams line up with
    // inference-time streams.
    static void append_query_runs(std::string_view token, std::vector<std::string>& runs) {
        std::string current;
        for (const unsigned char ch : token) {
            if (std::isalnum(ch) != 0) {
                current.push_back(static_cast<char>(std::tolower(ch)));
            } else if (!current.empty()) {
                runs.push_back(current);
                current.clear();
            }
        }
        if (!current.empty()) {
            runs.push_back(current);
        }
    }

    // Context tails hash one entry per lowercased alnum run — exactly the
    // convention lexical_tail applies to raw text — so trained tails, the
    // prompt-seeded tail, and the generation-time tail all live in one
    // space. (Hashing whole tokens used to split conventions: "a_b" hashed
    // as "ab" on one side but as "a","b" on the other.)
    static std::vector<std::uint64_t> context_tail_hashes(
        const std::vector<std::string>& context_tokens) {
        constexpr std::size_t tail_limit = 6;
        std::vector<std::uint64_t> tail;
        std::vector<std::string> runs;
        for (const auto& token : context_tokens) {
            if (is_subword_continuation(token)) {
                continue;
            }
            runs.clear();
            append_query_runs(token, runs);
            for (const auto& run : runs) {
                if (run.size() <= 1) {
                    continue;
                }
                tail.push_back(stable_hash(run));
                if (tail.size() > tail_limit) {
                    tail.erase(tail.begin());
                }
            }
        }
        return tail;
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

    static dz_real tail_overlap(const std::vector<std::uint64_t>& active_tail,
                                    const std::vector<std::uint64_t>& prototype_tail) {
        if (active_tail.empty() || prototype_tail.empty()) {
            return 0.0;
        }
        dz_real score = 0.0;
        dz_real denom = 0.0;
        for (std::size_t ai = 0; ai < active_tail.size(); ++ai) {
            const std::size_t active_recency = active_tail.size() - 1U - ai;
            const dz_real base_weight = 1.0 / (1.0 + static_cast<dz_real>(active_recency));
            denom += base_weight;
            dz_real best = 0.0;
            for (std::size_t pi = 0; pi < prototype_tail.size(); ++pi) {
                if (active_tail[ai] != prototype_tail[pi]) {
                    continue;
                }
                const std::size_t proto_recency = prototype_tail.size() - 1U - pi;
                const auto distance = active_recency > proto_recency
                                          ? active_recency - proto_recency
                                          : proto_recency - active_recency;
                best = std::max(best,
                                base_weight / (1.0 + 0.5 * static_cast<dz_real>(distance)));
            }
            score += best;
        }
        return denom <= 1.0e-30 ? 0.0 : std::clamp<dz_real>(score / denom, 0.0, 1.0);
    }

    dz_real random_unit() const {
        return std::generate_canonical<dz_real, std::numeric_limits<dz_real>::digits>(rng_);
    }

    dz_real centered_noise() const {
        return 2.0 * random_unit() - 1.0;
    }

    void add_complex_noise(std::vector<cx>& values, dz_real scale) const {
        if (scale <= 0.0) {
            return;
        }
        for (auto& value : values) {
            value += cx(scale * centered_noise(), scale * centered_noise());
        }
        normalize_complex(values);
    }

    void add_real_noise(std::vector<dz_real>& values, dz_real scale) const {
        if (scale <= 0.0) {
            return;
        }
        for (auto& value : values) {
            value += scale * centered_noise();
        }
        normalize_real(values);
    }

    // Returns how well the PRE-update memory already predicted this context
    // (|<key, context>|), so the caller can drive error-proportional
    // contrastive pressure.
    dz_real update_oscillator(TokenOscillator& oscillator,
                           const std::vector<cx>& target_key,
                           const std::vector<cx>& target_query,
                           const std::vector<cx>& target_transition,
                           const std::vector<dz_real>& target_padic,
                           const std::vector<dz_real>& target_query_padic,
                           const std::vector<std::uint64_t>& context_tail) {
        const dz_real before_loss =
            0.5 * complex_loss(oscillator.key, target_key) +
            0.5 * complex_loss(oscillator.query, target_query);
        // Surprise-gated (delta-rule flavored) learning: contexts the memory
        // already predicts barely move it — consolidation against wash-out by
        // frequent contexts — while novel contexts learn at full rate.
        const dz_real predicted = normalized_complex_similarity(oscillator.key, target_key);
        const dz_real surprise = std::clamp<dz_real>(1.0 - predicted, 0.0, 1.0);
        // The gate can exceed 1: a frequent token binding a genuinely NOVEL
        // context learns above base rate (cap 1.6), while well-predicted
        // repeats idle at the floor.
        const dz_real surprise_gate =
            std::clamp<dz_real>(surprise_floor_ + 1.60 * surprise, surprise_floor_, 1.60);
        const dz_real rate_base =
            learning_rate_ / std::sqrt(1.0 + 0.02 * static_cast<dz_real>(oscillator.observations));
        const dz_real rate = oscillator.observations == 0
                                     ? 1.0
                                     : std::min<dz_real>(1.0, surprise_gate * rate_base);
        for (std::size_t j = 0; j < dim_; ++j) {
            oscillator.key[j] = (1.0 - rate) * oscillator.key[j] + rate * target_key[j];
            oscillator.query[j] = (1.0 - rate) * oscillator.query[j] + rate * target_query[j];
            oscillator.transition[j] = (1.0 - rate) * oscillator.transition[j] + rate * target_transition[j];
            oscillator.padic_signature[j] =
                (1.0 - rate) * oscillator.padic_signature[j] + rate * target_padic[j];
            oscillator.query_padic_signature[j] =
                (1.0 - rate) * oscillator.query_padic_signature[j] + rate * target_query_padic[j];
        }
        normalize_complex(oscillator.key);
        normalize_complex(oscillator.query);
        normalize_complex(oscillator.transition);
        normalize_real(oscillator.padic_signature);
        normalize_real(oscillator.query_padic_signature);

        std::size_t best_prototype = std::numeric_limits<std::size_t>::max();
        dz_real best_match = -1.0;
        for (std::size_t i = 0; i < oscillator.prototypes.size(); ++i) {
            const auto& prototype = oscillator.prototypes[i];
            const dz_real spectral_match = normalized_complex_similarity(prototype.key, target_key);
            const dz_real padic_match = 0.5 + 0.5 * normalized_cosine(prototype.padic_signature, target_padic);
            const dz_real match = 0.82 * spectral_match + 0.18 * padic_match;
            if (match > best_match) {
                best_match = match;
                best_prototype = i;
            }
        }

        dz_real assigned_loss = before_loss;
        if (best_prototype == std::numeric_limits<std::size_t>::max() ||
            (best_match < 0.74 && oscillator.prototypes.size() < max_prototypes_per_token_)) {
            oscillator.prototypes.push_back({target_key,
                                             target_query,
                                             target_transition,
                                             target_padic,
                                             target_query_padic,
                                             std::vector<cx>(dim_, cx(0, 0)),
                                             std::vector<dz_real>(dim_, 0.0),
                                             context_tail,
                                             before_loss,
                                             1});
        } else {
            auto& prototype = oscillator.prototypes[best_prototype];
            const dz_real proto_loss =
                0.5 * complex_loss(prototype.key, target_key) +
                0.5 * complex_loss(prototype.query, target_query);
            // Prototype-level credit assignment: the gate follows how well
            // THIS prototype predicts the context, not the top-level key's
            // surprise (the correct signal for multi-context tokens).
            const dz_real proto_gate =
                std::clamp<dz_real>(surprise_floor_ + 1.60 * (1.0 - best_match),
                                    surprise_floor_,
                                    1.60);
            const dz_real proto_rate =
                prototype.observations == 0
                    ? 1.0
                    : std::min<dz_real>(0.65, rate_base * 1.25 * proto_gate);
            for (std::size_t j = 0; j < dim_; ++j) {
                prototype.key[j] = (1.0 - proto_rate) * prototype.key[j] + proto_rate * target_key[j];
                prototype.query[j] = (1.0 - proto_rate) * prototype.query[j] + proto_rate * target_query[j];
                prototype.transition[j] =
                    (1.0 - proto_rate) * prototype.transition[j] + proto_rate * target_transition[j];
                prototype.padic_signature[j] =
                    (1.0 - proto_rate) * prototype.padic_signature[j] + proto_rate * target_padic[j];
                prototype.query_padic_signature[j] =
                    (1.0 - proto_rate) * prototype.query_padic_signature[j] + proto_rate * target_query_padic[j];
            }
            normalize_complex(prototype.key);
            normalize_complex(prototype.query);
            normalize_complex(prototype.transition);
            normalize_real(prototype.padic_signature);
            normalize_real(prototype.query_padic_signature);
            ++prototype.observations;
            prototype.error_ema = 0.90 * prototype.error_ema + 0.10 * proto_loss;
            // Credited error: the responsible prototype absorbs the error,
            // so multi-context tokens (syntax glue) are not billed the
            // mixture loss of contexts their prototypes already separate.
            assigned_loss = std::min(before_loss, proto_loss);
        }

        ++oscillator.observations;
        ++total_observations_;
        oscillator.error_ema = oscillator.observations == 1
                                   ? before_loss
                                   : 0.92 * oscillator.error_ema + 0.08 * assigned_loss;
        oscillator.strength = std::clamp<dz_real>(0.35 + std::log1p(static_cast<dz_real>(oscillator.observations)) /
                                                    (1.0 + oscillator.error_ema),
                                         0.10,
                                         8.0);
        loss_ema_ = loss_updates_ == 0 ? before_loss : 0.98 * loss_ema_ + 0.02 * before_loss;
        ++loss_updates_;
        return predicted;
    }

    void update_contrastive_negatives(std::size_t positive_index,
                                      const std::vector<cx>& target_key,
                                      const std::vector<dz_real>& target_padic,
                                      dz_real own_match = 0.0) {
        if (oscs_.size() < 2) {
            return;
        }
        if (contrastive_period_ > 1 && total_observations_ % contrastive_period_ != 0) {
            return;
        }

        struct HardNegative {
            dz_real score;
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
                    const dz_real spectral_match = normalized_complex_similarity(key, target_key);
                    const dz_real padic_match = 0.5 + 0.5 * normalized_cosine(padic, target_padic);
                    const dz_real score = 0.84 * spectral_match + 0.16 * padic_match;
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

        // Total order: worker threads merge their candidate lists in
        // completion order, so a bare score comparison would let exact ties
        // resolve differently between runs and thread counts.
        const auto by_score = [](const HardNegative& left, const HardNegative& right) {
            if (left.score != right.score) {
                return left.score > right.score;
            }
            if (left.oscillator != right.oscillator) {
                return left.oscillator < right.oscillator;
            }
            return left.prototype < right.prototype;
        };
        const std::size_t take = std::min(max_hard_negatives_, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end(), by_score);
        for (std::size_t idx = 0; idx < take; ++idx) {
            auto& oscillator = oscs_[candidates[idx].oscillator];
            // Error-driven pressure (perceptron flavor): a competitor that
            // matches this context BETTER than the true token's own memory is
            // repelled hard; competitors the true token already dominates get
            // only residual pressure.
            const dz_real violation =
                std::max<dz_real>(0.0, candidates[idx].score - own_match);
            const dz_real error_gain =
                std::clamp<dz_real>(0.25 + 2.0 * violation, 0.25, 1.6);
            const dz_real rate = std::clamp<dz_real>(
                contrastive_rate_ * candidates[idx].score * error_gain, 0.005, 0.14);
            mix_negative_key(oscillator.negative_key, target_key, rate);
            mix_negative_padic(oscillator.negative_padic_signature, target_padic, rate);
            repel_from(oscillator.key, target_key, rate * 0.35);
            if (candidates[idx].prototype != no_prototype &&
                candidates[idx].prototype < oscillator.prototypes.size()) {
                auto& prototype = oscillator.prototypes[candidates[idx].prototype];
                mix_negative_key(prototype.negative_key, target_key, rate);
                mix_negative_padic(prototype.negative_padic_signature, target_padic, rate);
                repel_from(prototype.key, target_key, rate * 0.45);
            }
            ++contrastive_updates_;
        }
    }

    void drop_one() {
        std::size_t wi = 0; dz_real ws = 1e18;
        for (std::size_t i = 0; i < oscs_.size(); ++i) {
            dz_real s = oscs_[i].observations;
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
    std::size_t max_osc_;
    std::size_t dim_;
    std::size_t thread_count_;
    std::size_t parallel_min_dimensions_;
    std::vector<std::size_t> steps_;
    std::vector<dz_real> zeta_basis_;
    std::vector<std::uint32_t> seed_primes_;
    std::vector<dz_real> seed_theta_;
    std::vector<dz_real> seed_energy_;
    std::vector<dz_real> seed_padic_log_;
    std::vector<dz_real> seed_prime_phase_;
    mutable std::mt19937_64 rng_;
    mutable std::unique_ptr<RangeThreadPool> range_pool_;
    dz_real learning_rate_ = 0.32;
    dz_real generation_temperature_ = 0.08;
    dz_real contrastive_rate_ = 0.08;
    dz_real contrastive_margin_ = 0.62;
    dz_real contrastive_strength_ = 0.74;
    dz_real update_probability_ = 1.0;
    dz_real update_noise_ = 0.0;
    dz_real random_init_scale_ = 0.0;
    dz_real dimension_interference_ = 0.0;
    dz_real contrast_beta_ = 1.25;
    dz_real contrast_floor_ = 0.15;
    dz_real surprise_floor_ = 0.22;
    std::size_t max_prototypes_per_token_ = 4;
    std::size_t max_hard_negatives_ = 4;
    std::size_t max_context_tokens_ = 24;
    std::size_t contrastive_period_ = 3;
    std::size_t total_observations_ = 0;
    std::size_t contrastive_updates_ = 0;
    std::size_t loss_updates_ = 0;
    dz_real loss_ema_ = 0.0;
};

} // namespace dzeta
