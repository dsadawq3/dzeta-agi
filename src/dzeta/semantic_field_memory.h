#pragma once

#include "variational_core.h"
#include "mentalese_core.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct FieldMemoryAttractor {
    std::uint64_t id = 0;
    std::vector<long double> impulse_signature;
    std::vector<long double> basin_center;
    std::vector<long double> phase_trajectory;
    long double survival_energy = 0.0L;
    long double strength = 0.0L;
    long double entropy_penalty = 0.0L;
    bool survived = false;
    std::string prompt_digest;
    std::string output;
};

struct FieldMemoryHit {
    std::size_t index = 0;
    long double score = 0.0L;
    long double basin_similarity = 0.0L;
    long double impulse_similarity = 0.0L;
    std::size_t word_overlap = 0;
    std::string output;
};

inline std::size_t semantic_word_overlap(std::string_view left, std::string_view right) {
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

inline std::vector<long double> field_basin_center(const FieldState& field, std::size_t width = 32) {
    std::vector<long double> center(width, 0.0L);
    if (field.empty() || width == 0) {
        return center;
    }
    for (std::size_t i = 0; i < field.size(); ++i) {
        const std::size_t bucket = static_cast<std::size_t>(field.primes[i] % width);
        center[bucket] += std::cos(field.phases[i]) * (0.25L + field.activations[i]) +
                          0.25L * field.semantic_charge[i];
    }
    const long double norm = std::sqrt(std::inner_product(center.begin(), center.end(), center.begin(), 0.0L));
    if (norm > 1.0e-18L) {
        for (auto& item : center) {
            item /= norm;
        }
    }
    return center;
}

class SemanticFieldMemory {
public:
    std::size_t imprint(std::string_view prompt,
                        const FieldState& field,
                        const FieldEnergyReport& energy,
                        std::string output,
                        bool survived) {
        const auto impulse = field_impulse_signature(prompt, 32);
        const auto basin = field_basin_center(field, 32);
        const auto id = stable_hash(prompt) ^ (field_geometry_signature(field) << 1U);
        for (std::size_t i = 0; i < attractors_.size(); ++i) {
            auto& existing = attractors_[i];
            const auto basin_similarity = field_cosine_similarity(existing.basin_center, basin);
            const auto impulse_similarity = field_cosine_similarity(existing.impulse_signature, impulse);
            if (basin_similarity > 0.92L || (existing.id == id && impulse_similarity > 0.75L)) {
                existing.strength = std::clamp(existing.strength + (survived ? 0.18L : -0.08L),
                                               0.0L, 8.0L);
                existing.entropy_penalty = std::clamp(existing.entropy_penalty +
                                                      (survived ? -0.04L : 0.16L), 0.0L, 4.0L);
                existing.survival_energy = std::min(existing.survival_energy, energy.total_free_energy);
                existing.survived = existing.survived || survived;
                if (!output.empty()) {
                    existing.output = std::move(output);
                }
                return i;
            }
        }

        FieldMemoryAttractor attractor;
        attractor.id = id;
        attractor.impulse_signature = impulse;
        attractor.basin_center = basin;
        attractor.survival_energy = energy.total_free_energy;
        attractor.strength = survived ? 1.0L + energy.stability : 0.25L;
        attractor.entropy_penalty = survived ? 0.0L : 0.4L;
        attractor.survived = survived;
        attractor.prompt_digest = std::to_string(stable_hash(prompt));
        attractor.output = std::move(output);
        attractors_.push_back(std::move(attractor));
        return attractors_.size() - 1U;
    }

    std::vector<FieldMemoryHit> resonate(std::string_view prompt,
                                         const FieldState& field,
                                         std::size_t limit = 4) const {
        std::vector<FieldMemoryHit> hits;
        const auto impulse = field_impulse_signature(prompt, 32);
        const auto basin = field_basin_center(field, 32);
        for (std::size_t i = 0; i < attractors_.size(); ++i) {
            const auto& attractor = attractors_[i];
            FieldMemoryHit hit;
            hit.index = i;
            hit.basin_similarity = 0.5L + 0.5L * field_cosine_similarity(attractor.basin_center, basin);
            hit.impulse_similarity = 0.5L + 0.5L * field_cosine_similarity(attractor.impulse_signature, impulse);
            hit.word_overlap = semantic_word_overlap(prompt, attractor.output);
            hit.score = 0.52L * hit.basin_similarity +
                        0.30L * hit.impulse_similarity +
                        0.14L * std::clamp(attractor.strength / 4.0L, 0.0L, 1.0L) -
                        0.10L * std::clamp(attractor.entropy_penalty / 2.0L, 0.0L, 1.0L);
            hit.output = attractor.output;
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

    std::size_t size() const noexcept {
        return attractors_.size();
    }

    const std::vector<FieldMemoryAttractor>& attractors() const noexcept {
        return attractors_;
    }

    FieldMemoryAttractor& at(std::size_t index) {
        return attractors_.at(index);
    }

    const FieldMemoryAttractor& at(std::size_t index) const {
        return attractors_.at(index);
    }

    long double total_entropy_penalty() const {
        long double total = 0.0L;
        for (const auto& attractor : attractors_) {
            total += attractor.entropy_penalty;
        }
        return total;
    }

    FieldState pattern_complete(const FieldState& incomplete, std::size_t k = 1) const {
        if (attractors_.empty() || incomplete.empty()) {
            return incomplete;
        }
        const auto incomplete_sig = field_geometry_signature(incomplete);
        auto incomplete_mental = encode_prompt_mental_state(
            "pattern " + std::to_string(incomplete_sig), 32);
        std::vector<std::pair<std::size_t, long double>> scored;
        for (std::size_t i = 0; i < attractors_.size(); ++i) {
            const auto& a = attractors_[i];
            auto mem_mental = encode_prompt_mental_state(
                a.prompt_digest + " " + a.output, 32);
            const long double sim = mental_similarity(incomplete_mental, mem_mental);
            scored.emplace_back(i, sim);
        }
        std::sort(scored.begin(), scored.end(), [](const auto& x, const auto& y) {
            return x.second > y.second;
        });
        if (scored.empty() || scored[0].second < 0.15L) {
            return incomplete;
        }
        FieldState result = incomplete;
        for (std::size_t rank = 0; rank < std::min(k, scored.size()); ++rank) {
            const auto& best = attractors_[scored[rank].first];
            if (best.basin_center.empty()) continue;
            for (std::size_t i = 0; i < result.size(); ++i) {
                if (result.activations[i] < 0.01L) {
                    const std::size_t bucket = static_cast<std::size_t>(
                        result.primes[i] % best.basin_center.size());
                    result.activations[i] = std::clamp(
                        std::abs(best.basin_center[bucket]) * 0.6L + 0.2L, 0.0L, 1.0L);
                    result.phases[i] = wrap_phase(
                        result.phases[i] + best.basin_center[bucket] * 0.15L);
                }
            }
        }
        normalize_field_state(result);
        return result;
    }

    FieldState analogical_transfer(const FieldState& source, const FieldState& target) const {
        if (attractors_.empty()) return target;
        const auto source_sig = field_geometry_signature(source);
        auto source_mental = encode_prompt_mental_state(
            "analog " + std::to_string(source_sig), 32);
        std::size_t best_idx = 0;
        long double best_sim = 0.0L;
        for (std::size_t i = 0; i < attractors_.size(); ++i) {
            const auto& a = attractors_[i];
            auto mem_mental = encode_prompt_mental_state(
                a.prompt_digest + " " + a.output, 32);
            const long double sim = mental_similarity(source_mental, mem_mental);
            if (sim > best_sim) {
                best_sim = sim;
                best_idx = i;
            }
        }
        if (best_sim < 0.15L) return target;
        const auto& best = attractors_[best_idx];
        FieldState result = target;
        const std::size_t copy_count = std::min(best.basin_center.size(), result.size());
        for (std::size_t i = 0; i < copy_count; ++i) {
            result.phases[i] = wrap_phase(result.phases[i] + best.basin_center[i] * 0.3L);
            result.activations[i] = std::clamp(
                result.activations[i] + std::abs(best.basin_center[i]) * 0.4L, 0.0L, 1.0L);
            result.semantic_charge[i] = std::clamp(
                result.semantic_charge[i] + best.basin_center[i] * 0.2L, -1.0L, 1.0L);
        }
        normalize_field_state(result);
        return result;
    }

    std::size_t concept_formation() {
        if (attractors_.size() < 2) return 0;
        std::size_t concepts_formed = 0;
        std::vector<bool> used(attractors_.size(), false);
        for (std::size_t i = 0; i < attractors_.size(); ++i) {
            if (used[i]) continue;
            std::vector<std::size_t> group;
            group.push_back(i);
            for (std::size_t j = i + 1; j < attractors_.size(); ++j) {
                if (used[j]) continue;
                if (attractors_[i].basin_center.empty() || attractors_[j].basin_center.empty()) continue;
                const long double coherence = field_cosine_similarity(
                    attractors_[i].basin_center, attractors_[j].basin_center);
                if (coherence > 0.8L) {
                    group.push_back(j);
                }
            }
            if (group.size() >= 2U) {
                for (const auto idx : group) used[idx] = true;
                FieldMemoryAttractor concept_attr;
                concept_attr.id = stable_hash("concept_" + std::to_string(concepts_formed));
                concept_attr.basin_center = attractors_[group[0]].basin_center;
                for (std::size_t m = 1; m < group.size(); ++m) {
                    const auto& member = attractors_[group[m]];
                    for (std::size_t d = 0; d < std::min(concept_attr.basin_center.size(), member.basin_center.size()); ++d) {
                        concept_attr.basin_center[d] = (concept_attr.basin_center[d] *
                            static_cast<long double>(m) + member.basin_center[d]) /
                            static_cast<long double>(m + 1);
                    }
                }
                concept_attr.impulse_signature = attractors_[group[0]].impulse_signature;
                concept_attr.strength = 1.5L;
                concept_attr.survived = true;
                concept_attr.prompt_digest = "concept_" + std::to_string(concepts_formed);
                concept_attr.output = "concept of " + std::to_string(group.size()) + " coherent patterns";
                attractors_.push_back(std::move(concept_attr));
                ++concepts_formed;
            }
        }
        return concepts_formed;
    }

private:
    std::vector<FieldMemoryAttractor> attractors_;

    friend void save_semantic_field_memory(const SemanticFieldMemory&, std::string_view);
    friend SemanticFieldMemory load_semantic_field_memory(std::string_view);
};

inline void write_vector(std::ostream& out, const std::vector<long double>& values) {
    out << values.size();
    for (auto value : values) {
        out << ',' << static_cast<double>(value);
    }
}

inline std::vector<long double> read_vector(std::string_view text) {
    std::stringstream stream{std::string(text)};
    std::string item;
    std::getline(stream, item, ',');
    const auto count = static_cast<std::size_t>(std::stoull(item));
    std::vector<long double> values;
    values.reserve(count);
    while (std::getline(stream, item, ',')) {
        values.push_back(std::stold(item));
    }
    values.resize(count, 0.0L);
    return values;
}

inline std::string semantic_memory_hex_encode(std::string_view text) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(4U + text.size() * 2U);
    out += "hex:";
    for (unsigned char ch : text) {
        out.push_back(hex[(ch >> 4U) & 0x0FU]);
        out.push_back(hex[ch & 0x0FU]);
    }
    return out;
}

inline int semantic_memory_hex_value(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return 10 + ch - 'a';
    }
    if (ch >= 'A' && ch <= 'F') {
        return 10 + ch - 'A';
    }
    return -1;
}

inline std::string semantic_memory_hex_decode(std::string_view text) {
    if (text.rfind("hex:", 0) != 0) {
        return std::string(text);
    }
    text.remove_prefix(4);
    std::string out;
    out.reserve(text.size() / 2U);
    for (std::size_t i = 0; i + 1U < text.size(); i += 2U) {
        const int hi = semantic_memory_hex_value(text[i]);
        const int lo = semantic_memory_hex_value(text[i + 1U]);
        if (hi < 0 || lo < 0) {
            return {};
        }
        out.push_back(static_cast<char>((hi << 4U) | lo));
    }
    return out;
}

inline void save_semantic_field_memory(const SemanticFieldMemory& memory, std::string_view path) {
    std::ofstream output(std::string(path), std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write semantic field memory: " + std::string(path));
    }
    output << "DZETA_SEMANTIC_FIELD_MEMORY_V1\n";
    for (const auto& attractor : memory.attractors_) {
        output << attractor.id << '\t'
               << static_cast<double>(attractor.survival_energy) << '\t'
               << static_cast<double>(attractor.strength) << '\t'
               << static_cast<double>(attractor.entropy_penalty) << '\t'
               << (attractor.survived ? 1 : 0) << '\t'
               << attractor.prompt_digest << '\t';
        write_vector(output, attractor.impulse_signature);
        output << '\t';
        write_vector(output, attractor.basin_center);
        output << '\t' << semantic_memory_hex_encode(attractor.output) << '\n';
    }
}

inline SemanticFieldMemory load_semantic_field_memory(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read semantic field memory: " + std::string(path));
    }
    std::string header;
    std::getline(input, header);
    if (header != "DZETA_SEMANTIC_FIELD_MEMORY_V1") {
        throw std::runtime_error("invalid semantic field memory header: " + std::string(path));
    }
    SemanticFieldMemory memory;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields;
        std::size_t begin = 0;
        for (;;) {
            const auto tab = line.find('\t', begin);
            if (tab == std::string::npos) {
                fields.push_back(line.substr(begin));
                break;
            }
            fields.push_back(line.substr(begin, tab - begin));
            begin = tab + 1;
        }
        if (fields.size() != 9) {
            throw std::runtime_error("invalid semantic field memory row: " + std::string(path));
        }
        FieldMemoryAttractor attractor;
        attractor.id = static_cast<std::uint64_t>(std::stoull(fields[0]));
        attractor.survival_energy = std::stold(fields[1]);
        attractor.strength = std::stold(fields[2]);
        attractor.entropy_penalty = std::stold(fields[3]);
        attractor.survived = fields[4] == "1";
        attractor.prompt_digest = fields[5];
        attractor.impulse_signature = read_vector(fields[6]);
        attractor.basin_center = read_vector(fields[7]);
        attractor.output = semantic_memory_hex_decode(fields[8]);
        memory.attractors_.push_back(std::move(attractor));
    }
    return memory;
}

} // namespace dzeta
