#pragma once

#include "dzeta_vm.h"
#include "field_state.h"
#include "sat.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct MentalToken {
    std::string symbol;
    std::vector<long double> vector;
    long double salience = 0.0L;
};

struct MentalState {
    std::vector<long double> vectors;
    std::vector<std::string> symbols;
    long double uncertainty = 0.5L;
    std::vector<std::string> goals;
    std::uint64_t trace_id = 0;

    bool has_symbol(std::string_view symbol) const {
        return std::find(symbols.begin(), symbols.end(), symbol) != symbols.end();
    }
};

struct MentalTrace {
    std::uint64_t trace_id = 0;
    MentalState initial_state;
    MentalState final_state;
    std::vector<std::string> modules_used;
    std::vector<std::string> events;
    long double free_energy = 1.0L;
    long double reward = 0.0L;

    bool contains(std::string_view text) const {
        for (const auto& event : events) {
            if (event.find(text) != std::string::npos) {
                return true;
            }
        }
        for (const auto& module : modules_used) {
            if (module.find(text) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
};

struct LatentProgram {
    std::string name;
    BytecodeProgram bytecode;
    MentalState signature;
};

struct ThoughtModule {
    std::string name;
    std::string input_signature;
    std::string output_signature;
    BytecodeProgram program;
    std::vector<long double> trainable_weights;
    bool enabled = true;
};

inline void mental_add_symbol(MentalState& state, std::string symbol) {
    if (!state.has_symbol(symbol)) {
        state.symbols.push_back(std::move(symbol));
    }
}

inline std::vector<long double> mental_vector_from_text(std::string_view text, std::size_t dimension = 32) {
    std::vector<long double> vector(std::max<std::size_t>(1, dimension), 0.0L);
    const auto tokens = tokenize_query(text);
    if (tokens.empty()) {
        return vector;
    }
    for (std::size_t token_index = 0; token_index < tokens.size(); ++token_index) {
        std::uint64_t seed = stable_hash(tokens[token_index]) ^ (0x9e3779b97f4a7c15ULL * (token_index + 1U));
        for (std::size_t i = 0; i < vector.size(); ++i) {
            vector[i] += 2.0L * field_unit_from_hash(splitmix64(seed)) - 1.0L;
        }
    }
    const long double norm = std::sqrt(std::inner_product(vector.begin(), vector.end(), vector.begin(), 0.0L));
    if (norm > 1.0e-12L) {
        for (auto& value : vector) {
            value /= norm;
        }
    }
    return vector;
}

inline void infer_structural_symbols(MentalState& state, std::string_view text) {
    const auto tokens = tokenize_query(text);
    std::size_t ordering_markers = 0;
    std::size_t code_markers = 0;
    std::size_t numeric_markers = 0;
    for (const auto& token : tokens) {
        if (token == "first" || token == "then" || token == "next" || token == "after" || token == "step") {
            ++ordering_markers;
        }
        if (token == "def" || token == "return" || token == "class" || token == "function") {
            ++code_markers;
        }
        if (std::any_of(token.begin(), token.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
            ++numeric_markers;
        }
    }
    if (ordering_markers >= 2) {
        mental_add_symbol(state, "ordered_process");
    }
    if (code_markers != 0) {
        mental_add_symbol(state, "code_task");
    }
    if (numeric_markers != 0 || text.find("->") != std::string_view::npos || text.find("task:") != std::string_view::npos) {
        mental_add_symbol(state, "example_task");
    }
    if (text.find("why") != std::string_view::npos || text.find("cause") != std::string_view::npos) {
        mental_add_symbol(state, "causal_query");
    }
}

inline MentalState encode_prompt_mental_state(std::string_view prompt, std::size_t dimension = 32) {
    MentalState state;
    state.vectors = mental_vector_from_text(prompt, dimension);
    state.trace_id = stable_hash(prompt);
    state.goals.push_back(std::string(prompt));
    state.uncertainty = std::clamp(1.0L / (1.0L + static_cast<long double>(tokenize_query(prompt).size())),
                                   0.05L, 0.95L);
    mental_add_symbol(state, "prompt");
    infer_structural_symbols(state, prompt);
    const auto tokens = tokenize_query(prompt);
    std::size_t salient = 0;
    for (const auto& token : tokens) {
        if (token.size() < 4 || token == "with" || token == "from" || token == "that" || token == "this") {
            continue;
        }
        mental_add_symbol(state, "token:" + token);
        if (++salient >= 6U) {
            break;
        }
    }
    return state;
}

inline MentalState encode_program_mental_state(const VmProgram& program, std::size_t dimension = 32) {
    auto state = encode_prompt_mental_state("program " + program.name + " " + vm_program_kind_name(program.kind), dimension);
    mental_add_symbol(state, "program:" + vm_program_kind_name(program.kind));
    state.uncertainty = 0.18L;
    return state;
}

inline MentalState encode_bytecode_program_mental_state(const BytecodeProgram& program, std::size_t dimension = 32) {
    auto state = encode_prompt_mental_state("bytecode " + program.name, dimension);
    mental_add_symbol(state, "bytecode:" + program.name);
    state.uncertainty = 0.16L;
    return state;
}

template <typename EpisodeLike>
MentalState encode_episode_mental_state(const EpisodeLike& episode, std::size_t dimension = 32) {
    auto state = encode_prompt_mental_state(episode.timestamp + " " + episode.prompt + " " + episode.answer, dimension);
    mental_add_symbol(state, "episode:user_event");
    state.uncertainty = std::clamp(1.0L - episode.confidence, 0.05L, 0.95L);
    return state;
}

inline MentalState encode_causal_variable_mental_state(std::string_view cause,
                                                       std::string_view effect,
                                                       std::size_t dimension = 32) {
    auto state = encode_prompt_mental_state(std::string(cause) + " causes " + std::string(effect), dimension);
    mental_add_symbol(state, "cause:" + std::string(cause));
    mental_add_symbol(state, "effect:" + std::string(effect));
    state.uncertainty = 0.25L;
    return state;
}

inline long double mental_similarity(const MentalState& left, const MentalState& right) {
    const std::size_t n = std::min(left.vectors.size(), right.vectors.size());
    long double dot = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        dot += left.vectors[i] * right.vectors[i];
    }
    std::size_t shared = 0;
    for (const auto& symbol : left.symbols) {
        if (right.has_symbol(symbol)) {
            ++shared;
        }
    }
    const long double symbol_score = left.symbols.empty() && right.symbols.empty()
                                         ? 0.0L
                                         : static_cast<long double>(shared) /
                                               static_cast<long double>(std::max(left.symbols.size(), right.symbols.size()));
    return std::clamp(0.45L * ((dot + 1.0L) * 0.5L) + 0.55L * symbol_score, 0.0L, 1.0L);
}

} // namespace dzeta
