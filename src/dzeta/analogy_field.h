#pragma once

#include "latent_field_space.h"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct AnalogySignature {
    LatentVector latent;
    std::vector<long double> topology;
    std::vector<std::string> words;

    std::size_t word_overlap_with(const AnalogySignature& other) const {
        std::size_t overlap = 0;
        for (const auto& word : words) {
            if (std::find(other.words.begin(), other.words.end(), word) != other.words.end()) {
                ++overlap;
            }
        }
        return overlap;
    }
};

inline bool has_any_token(const std::vector<std::string>& words, std::initializer_list<std::string_view> wanted) {
    for (auto item : wanted) {
        if (std::find(words.begin(), words.end(), item) != words.end()) {
            return true;
        }
    }
    return false;
}

inline AnalogySignature make_analogy_signature(std::string_view task, const LatentFieldSpace& latent_space) {
    AnalogySignature signature;
    signature.latent = latent_space.encode(task);
    signature.words = tokenize_query(task);
    signature.topology.assign(8, 0.0L);
    const auto& w = signature.words;
    signature.topology[0] = std::min<long double>(1.0L, static_cast<long double>(w.size()) / 12.0L);
    signature.topology[1] = has_any_token(w, {"first", "then", "after", "step", "steps", "plan", "serve", "test", "install", "attach", "tighten", "cook", "heat", "crack"}) ? 1.0L : 0.0L;
    signature.topology[2] = has_any_token(w, {"use", "with", "tool", "pan", "wheel", "chain", "frame", "resource", "input"}) ? 1.0L : 0.0L;
    signature.topology[3] = has_any_token(w, {"finish", "serve", "test", "ride", "return", "output", "done"}) ? 1.0L : 0.0L;
    signature.topology[4] = has_any_token(w, {"if", "when", "until", "check", "verify", "tighten", "heat"}) ? 1.0L : 0.0L;
    signature.topology[5] = has_any_token(w, {"assemble", "build", "cook", "prepare", "make", "create", "install", "attach"}) ? 1.0L : 0.0L;
    signature.topology[6] = has_any_token(w, {"why", "cause", "because", "effect"}) ? 1.0L : 0.0L;
    signature.topology[7] = has_any_token(w, {"compare", "like", "similar", "analogy"}) ? 1.0L : 0.0L;
    return signature;
}

inline long double topology_similarity(const std::vector<long double>& left,
                                       const std::vector<long double>& right) {
    return 0.5L + 0.5L * field_cosine_similarity(left, right);
}

inline long double analogy_score(const AnalogySignature& left, const AnalogySignature& right) {
    const long double latent = 0.5L + 0.5L * latent_cosine(left.latent, right.latent);
    const long double topology = topology_similarity(left.topology, right.topology);
    const long double overlap_penalty = std::min<long double>(0.15L, 0.03L * static_cast<long double>(left.word_overlap_with(right)));
    return std::clamp(0.72L * topology + 0.28L * latent - overlap_penalty, 0.0L, 1.0L);
}

inline std::string describe_analogy(const AnalogySignature& left, const AnalogySignature& right) {
    const auto score = analogy_score(left, right);
    if (score > 0.45L && left.topology.size() > 5 && right.topology.size() > 5 &&
        left.topology[1] > 0.5L && right.topology[1] > 0.5L && left.topology[5] > 0.5L && right.topology[5] > 0.5L) {
        return "ordered assembly process";
    }
    return score > 0.45L ? "shared trajectory topology" : "weak analogy";
}

} // namespace dzeta
