#pragma once

#include "latent_field_space.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct ReconsolidationEntry {
    std::string prompt;
    LatentVector latent;
    std::string output;
    long double reliability = 0.0L;
    long double recency = 1.0L;
    long double contradiction = 0.0L;
    bool success = false;
    std::string recovery_path;
};

struct ReconsolidationHit {
    std::size_t index = 0;
    long double score = 0.0L;
    long double reliability_before = 0.0L;
    std::string output;
};

class ReconsolidationMemory {
public:
    std::size_t store(std::string prompt,
                      LatentVector latent,
                      std::string output,
                      long double reliability,
                      bool success) {
        ReconsolidationEntry entry;
        entry.prompt = std::move(prompt);
        entry.latent = std::move(latent);
        entry.output = std::move(output);
        entry.reliability = std::clamp(reliability, 0.0L, 1.0L);
        entry.success = success;
        entries_.push_back(std::move(entry));
        return entries_.size() - 1U;
    }

    void decay(std::size_t ticks = 1) {
        const long double factor = std::pow(0.985L, static_cast<long double>(ticks));
        for (auto& entry : entries_) {
            entry.recency *= factor;
            if (!entry.success || entry.reliability < 0.4L) {
                entry.reliability *= factor;
            } else {
                entry.reliability = std::clamp(entry.reliability * (0.995L + 0.005L * factor), 0.0L, 1.0L);
            }
        }
    }

    std::vector<ReconsolidationHit> retrieve(std::string_view prompt,
                                             const LatentVector& query,
                                             std::size_t limit = 4) const {
        std::vector<ReconsolidationHit> hits;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            const auto& entry = entries_[i];
            const long double similarity = 0.5L + 0.5L * latent_cosine(entry.latent, query);
            const long double prompt_overlap = static_cast<long double>(word_overlap(prompt, entry.prompt)) /
                                               static_cast<long double>(std::max<std::size_t>(1, tokenize_query(prompt).size()));
            ReconsolidationHit hit;
            hit.index = i;
            hit.reliability_before = entry.reliability;
            hit.score = 0.60L * similarity + 0.22L * entry.reliability +
                        0.12L * entry.recency + 0.06L * prompt_overlap -
                        0.08L * entry.contradiction;
            hit.output = entry.output;
            hits.push_back(std::move(hit));
        }
        std::sort(hits.begin(), hits.end(), [](const auto& left, const auto& right) {
            return left.score > right.score;
        });
        if (hits.size() > limit) {
            hits.resize(limit);
        }
        return hits;
    }

    void reconsolidate(std::size_t index, bool success, std::string recovery_path) {
        if (index >= entries_.size()) {
            return;
        }
        auto& entry = entries_[index];
        entry.success = entry.success || success;
        entry.recency = 1.0L;
        entry.reliability = std::clamp(entry.reliability + (success ? 0.18L : -0.12L), 0.0L, 1.0L);
        entry.contradiction = std::clamp(entry.contradiction + (success ? -0.08L : 0.10L), 0.0L, 1.0L);
        entry.recovery_path = std::move(recovery_path);
    }

    void add_contradiction(std::size_t left, std::size_t right) {
        if (left < entries_.size()) {
            entries_[left].contradiction = std::min(1.0L, entries_[left].contradiction + 0.25L);
        }
        if (right < entries_.size()) {
            entries_[right].contradiction = std::min(1.0L, entries_[right].contradiction + 0.25L);
        }
    }

    const std::vector<ReconsolidationEntry>& entries() const noexcept {
        return entries_;
    }

private:
    static std::size_t word_overlap(std::string_view left, std::string_view right) {
        const auto a = tokenize_query(left);
        const auto b = tokenize_query(right);
        std::size_t overlap = 0;
        for (const auto& token : a) {
            if (std::find(b.begin(), b.end(), token) != b.end()) {
                ++overlap;
            }
        }
        return overlap;
    }

    std::vector<ReconsolidationEntry> entries_;
};

} // namespace dzeta
