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
#include <numeric>
#include <set>
#include <string>
#include <string_view>
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
    explicit OscillatorField(std::size_t max_osc = 4096, std::size_t dim = 192)
        : max_osc_(std::max<std::size_t>(128, max_osc)), dim_(dim) {
        steps_.resize(dim_);
        std::size_t nz = zeta_zero_count();
        for (std::size_t z = 0; z < dim_; ++z) {
            if (z < 64) steps_[z] = 1;           // zeros 0-63, fine
            else if (z < 128) steps_[z] = 4;     // zeros 64-127, every 4th
            else steps_[z] = nz / dim_;           // rest
        }
        std::srand(static_cast<unsigned>(std::time(nullptr)));
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
            bool found = false;
            for (auto& o : oscs_) { if (o.token == t) { found = true; break; } }
            if (!found) {
                if (oscs_.size() >= max_osc_) drop_one();
                oscs_.push_back({std::move(t), std::vector<cx>(dim_, cx(0,0)),
                                 std::vector<cx>(dim_, cx(0,0)),
                                 std::vector<long double>(dim_, 0.0L), 1});
            }
        }
    }

    void learn(std::string_view text) {
        auto tokens = tokenize_code(text, std::min<std::size_t>(text.size(), 128));
        if (tokens.size() < 3) { embed(text); return; }
        embed(text);
        for (std::size_t ti = 0; ti + 1 < tokens.size(); ++ti) {
            if (bad_token(tokens[ti])) continue;
            std::string prefix, next_prefix;
            for (std::size_t j = 0; j <= ti + 1; ++j) {
                if (j == ti + 1) {
                    next_prefix = prefix;
                    if (j > 0 && !bad_token(tokens[j])) next_prefix += ' ';
                    if (!bad_token(tokens[j])) next_prefix += tokens[j];
                }
                if (j > 0 && !bad_token(tokens[j])) prefix += ' ';
                if (!bad_token(tokens[j])) prefix += tokens[j];
            }
            for (auto& o : oscs_) {
                if (o.token == tokens[ti] && o.observations == 1) {
                    auto f_pre = make_seed_field_state(prefix, 64);
                    auto [pre_ampl, _] = weyl_transform(f_pre);
                    auto f_full = make_seed_field_state(next_prefix, 64);
                    auto [full_ampl, padic] = weyl_transform(f_full);
                    if (std::all_of(pre_ampl.begin(), pre_ampl.end(), [](cx v) { return std::abs(v) < 1e-30L; })) break;
                    for (std::size_t j = 0; j < dim_; ++j)
                        o.key[j] = full_ampl[j] - pre_ampl[j];  // delta = change
                    o.query = full_ampl;    // where we GO
                    o.padic_signature = padic;
                    cx n = 0;
                    for (auto v : o.query) n += v * std::conj(v);
                    long double nn = std::sqrt(std::abs(n));
                    if (nn > 1e-30L) for (auto& v : o.query) v /= nn;
                    o.observations = 2;
                    break;
                }
            }
        }
    }

    std::string forward(std::string_view text, std::size_t max_tokens = 24) {
        auto field = make_seed_field_state(text, 64);
        auto fp = weyl_transform(field).first;
        std::string out;
        std::set<std::string> used;
        auto normalize = [&]() {
            cx n = 0; for (auto v : fp) n += v * std::conj(v);
            long double fn = std::sqrt(std::abs(n));
            if (fn > 1e-30L) for (auto& v : fp) v /= fn;
        };
        normalize();

        for (std::size_t s = 0; s < max_tokens; ++s) {
            for (std::size_t z = 0; z < dim_; ++z) {
                long double theta = static_cast<long double>(s) * 0.005L * (1.0L + static_cast<long double>(z) * 0.1L);
                fp[z] *= cx(std::cos(theta), std::sin(theta));
            }

            // compute raw delta matches
            std::vector<std::pair<long double, std::size_t>> raw;
            for (std::size_t i = 0; i < oscs_.size(); ++i) {
                if (used.find(oscs_[i].token) != used.end()) continue;
                if (oscs_[i].token.size() <= 1) continue;
                cx dm = 0;
                for (std::size_t j = 0; j < dim_; ++j)
                    dm += std::conj(oscs_[i].key[j] + oscs_[i].query[j]) * fp[j];
                long double s = std::abs(dm);
                if (s > 1e-12L) raw.emplace_back(s, i);
            }
            if (raw.empty()) break;
            std::partial_sort(raw.begin(), raw.begin() + std::min<std::size_t>(64, raw.size()),
                              raw.end(), std::greater<>());
            std::size_t K = std::min<std::size_t>(64, raw.size());
            long double max_s = raw[0].first;
            // lateral inhibition: suppress similar oscillators
            std::vector<std::pair<long double, std::size_t>> inhibited;
            for (std::size_t t = 0; t < K; ++t) {
                long double score = raw[t].first;
                auto& o = oscs_[raw[t].second];
                for (std::size_t u = 0; u < t; ++u) {
                    auto& ou = oscs_[raw[u].second];
                    cx cross = 0;
                    long double n1 = 0, n2 = 0;
                    for (std::size_t j = 0; j < dim_; ++j) {
                        cross += std::conj(o.query[j]) * ou.query[j];
                        n1 += std::abs(o.query[j]) * std::abs(o.query[j]);
                        n2 += std::abs(ou.query[j]) * std::abs(ou.query[j]);
                    }
                    long double sim = std::abs(cross) / (std::sqrt(n1 * n2) + 1e-30L);
                    score -= 0.3L * sim * raw[t].first;  // inhibition by similarity
                }
                inhibited.emplace_back(score, raw[t].second);
            }
            std::partial_sort(inhibited.begin(), inhibited.begin() + 1, inhibited.end(), std::greater<>());
            std::size_t best_i = inhibited[0].second;
            if (!out.empty()) out += ' ';
            out += oscs_[best_i].token;
            used.insert(oscs_[best_i].token);
            fp = oscs_[best_i].query;
            normalize();
        }
        return out;
    }

    std::size_t size() const noexcept { return oscs_.size(); }
    void clear() { oscs_.clear(); }

private:
    struct TokenOscillator {
        std::string token;
        std::vector<cx> query;    // next-state projection (where to go)
        std::vector<cx> key;      // current-state projection (where we are)
        std::vector<long double> padic_signature;
        std::size_t observations = 1;
    };

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
                long double phase_noise = 0.05L * static_cast<long double>(std::rand() % 1000) / 1000.0L;
                long double re = std::cos(theta * zr + charge * 0.5L + phase_noise);
                long double im = std::sin(theta * zr + charge * 0.5L + phase_noise);
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
        if (wi + 1 < oscs_.size()) oscs_[wi] = std::move(oscs_.back());
        oscs_.pop_back();
    }

    std::vector<TokenOscillator> oscs_;
    std::vector<cx> fp_cx_;
    std::size_t max_osc_;
    std::size_t dim_;
    std::vector<std::size_t> steps_;
};

} // namespace dzeta