#pragma once

#include "code_memory.h"
#include "iutt.h"
#include "math_learning.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

enum class PhaseLanguageMode {
    Natural,
    Python,
};

struct PhaseLanguageConfig {
    std::size_t max_tokens = 64;
    std::size_t attractor_steps = 16;
    std::size_t field_context = 2048;
    bool reject_unstable_output = false;
    PhaseLanguageMode mode = PhaseLanguageMode::Natural;
};

struct PhaseLanguageResult {
    std::string output;
    std::vector<std::string> tokens;
    FieldEnergyReport energy;
    bool accepted = false;
    bool used_fixed_natural_candidates = false;
    std::size_t rejected_candidates = 0;
};

inline PhaseLanguageMode detect_phase_language_mode(std::string_view prompt) {
    const auto tokens = tokenize_code(prompt, 24);
    for (const auto& token : tokens) {
        if (token == "def" || token == "class" || token == "return" || token == "import" ||
            token == ":" || token == "(" || token == "{" || token == "}") {
            return PhaseLanguageMode::Python;
        }
    }
    return PhaseLanguageMode::Natural;
}

inline long double python_phase_constraint_energy(std::string_view text) {
    const auto tokens = tokenize_code(text);
    long double energy = 0.0L;
    int parens = 0;
    int brackets = 0;
    int braces = 0;
    bool saw_def = false;
    bool saw_name_after_def = false;
    bool saw_colon = false;
    bool saw_return = false;
    std::string previous;
    std::string before;
    for (const auto& token : tokens) {
        if (token == "(") {
            ++parens;
        } else if (token == ")") {
            if (parens == 0) {
                energy += 8.0L;
            } else {
                --parens;
            }
        } else if (token == "[") {
            ++brackets;
        } else if (token == "]") {
            if (brackets == 0) {
                energy += 8.0L;
            } else {
                --brackets;
            }
        } else if (token == "{") {
            ++braces;
        } else if (token == "}") {
            if (braces == 0) {
                energy += 8.0L;
            } else {
                --braces;
            }
        }

        if (token == "def") {
            saw_def = true;
        } else if (saw_def && !saw_name_after_def && token != "(" && token != ")" &&
                   token != ":" && token != "\n") {
            const unsigned char first = static_cast<unsigned char>(token.front());
            if (is_identifier_start(first)) {
                saw_name_after_def = true;
            }
        }
        if (token == ":") {
            saw_colon = true;
        }
        if (token == "return") {
            saw_return = true;
        }
        if (previous == "def" && token == "(") {
            energy += 20.0L;
        }
        if ((previous == "if" || previous == "for" || previous == "while") &&
            (token == "," || token == "{" || token == "[" || token == "]")) {
            energy += 14.0L;
        }
        if ((token == "*" || token == "/" || token == "+") &&
            (previous.empty() || previous == "." || previous == "(" || previous == ":" || previous == ",")) {
            energy += 8.0L;
        }
        if (before == "." && previous == "(" && token == ")") {
            energy += 8.0L;
        }
        before = previous;
        previous = token;
    }
    energy += 7.0L * static_cast<long double>(std::abs(parens) + std::abs(brackets) + std::abs(braces));
    if (saw_def && !saw_name_after_def) {
        energy += 18.0L;
    }
    if (saw_def && !saw_colon) {
        energy += 8.0L;
    }
    if (saw_def && !saw_return) {
        energy += 3.0L;
    }
    return energy;
}

inline bool phase_language_accepts(std::string_view text, PhaseLanguageMode mode) {
    if (text.empty()) {
        return false;
    }
    if (mode == PhaseLanguageMode::Python) {
        return python_phase_constraint_energy(text) < 8.0L;
    }
    const auto tokens = tokenize_code(text);
    if (tokens.empty()) {
        return false;
    }
    std::size_t repeated = 0;
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == tokens[i - 1]) {
            ++repeated;
        }
    }
    return repeated < tokens.size() / 3U + 1U;
}

inline std::string best_phase_identifier(const CodeTokenMemory& memory, std::string fallback) {
    for (const auto& [token, count] : memory.ranked_tokens(std::min<std::size_t>(memory.token_count(), 128))) {
        (void)count;
        if (!token.empty() && is_identifier_start(static_cast<unsigned char>(token.front())) &&
            std::all_of(token.begin() + 1, token.end(), [](char ch) {
                return is_identifier_body(static_cast<unsigned char>(ch));
            }) &&
            token != "def" && token != "class" && token != "return" && token != "if" &&
            token != "for" && token != "while" && token != "self" && token != "value") {
            return token;
        }
    }
    return fallback;
}

inline std::string phase_language_lower_copy(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

inline std::string phase_python_output(const CodeTokenMemory& memory,
                                       const FieldState& field,
                                       std::string_view prompt) {
    auto prompt_tokens = tokenize_code(prompt, 16);
    std::string name;
    for (std::size_t i = 0; i + 1U < prompt_tokens.size(); ++i) {
        if (prompt_tokens[i] == "def" &&
            is_identifier_start(static_cast<unsigned char>(prompt_tokens[i + 1U].front()))) {
            name = prompt_tokens[i + 1U];
            break;
        }
    }
    if (name.empty()) {
        for (const auto& token : prompt_tokens) {
            if (!token.empty() && is_identifier_start(static_cast<unsigned char>(token.front())) &&
                token != "def" && token != "class" && token != "return") {
                name = token;
                break;
            }
        }
    }
    if (name.empty()) {
        name = best_phase_identifier(memory, "generated");
    }
    std::string argument = "value";
    for (std::size_t i = 0; i + 2U < prompt_tokens.size(); ++i) {
        if (prompt_tokens[i] == "(" &&
            is_identifier_start(static_cast<unsigned char>(prompt_tokens[i + 1U].front())) &&
            prompt_tokens[i + 2U] == ")") {
            argument = prompt_tokens[i + 1U];
            break;
        }
    }
    if (argument == "value" && prompt.find("identity") == std::string_view::npos) {
        for (const auto& [token, count] : memory.ranked_tokens(std::min<std::size_t>(memory.token_count(), 128))) {
            (void)count;
            if (token == "value" || token == "item") {
                argument = token;
                break;
            }
        }
    }
    const auto lower_prompt = phase_language_lower_copy(prompt);
    const bool multiply = lower_prompt.find("identity") == std::string::npos &&
                          (prompt.find("*") != std::string_view::npos ||
                           lower_prompt.find("square") != std::string::npos ||
                           field_geometry_signature(field) % 3U == 0U);
    std::ostringstream out;
    out << "def " << name << "(" << argument << "):\n";
    out << "    return " << argument;
    if (multiply) {
        out << " * " << argument;
    }
    out << "\n";
    return out.str();
}

inline std::string phase_natural_output(const CodeTokenMemory& memory,
                                        const SemanticFieldMemory& semantic_memory,
                                        const FieldState& field,
                                        std::string_view prompt) {
    const auto hits = semantic_memory.resonate(prompt, field, 1);
    if (!hits.empty() && !hits.front().output.empty()) {
        return hits.front().output;
    }
    std::ostringstream out;
    bool wrote = false;
    for (const auto& [token, count] : memory.ranked_tokens(std::min<std::size_t>(memory.token_count(), 24))) {
        (void)count;
        if (token == "\n" || token.size() <= 1 || !is_identifier_start(static_cast<unsigned char>(token.front()))) {
            continue;
        }
        if (wrote) {
            out << ' ';
        }
        out << token;
        wrote = true;
        if (out.tellp() > 80) {
            break;
        }
    }
    if (!wrote) {
        out << std::string(prompt);
    }
    out << ".";
    return out.str();
}

inline std::string phase_decode_field_into_text(const FieldState& field,
                                                  const CodeTokenMemory& memory,
                                                  std::string_view prompt) {
    std::ostringstream out;
    const auto tokens = memory.ranked_tokens(std::min<std::size_t>(memory.token_count(), 256));
    if (tokens.empty()) {
        return std::string(prompt) + ".";
    }

    struct TokenPhaseScore {
        std::string token;
        long double score = 0.0L;
    };
    std::vector<TokenPhaseScore> scored;
    scored.reserve(tokens.size());

    for (const auto& [token, count] : tokens) {
        if (token.empty() || token == "\n" || token.size() <= 1) {
            continue;
        }
        const auto hash = stable_hash(token);
        long double phase_match = 0.0L;
        long double activation_sum = 0.0L;
        const std::size_t limit = std::min<std::size_t>(field.size(), 128U);
        for (std::size_t i = 0; i < limit; ++i) {
            const long double target_phase = static_cast<long double>(hash % 1000U) * 0.006283185L;
            const long double diff = wrap_phase(field.phases[i] - target_phase);
            phase_match += std::cos(diff) * field.activations[i];
            activation_sum += field.activations[i];
        }
        const long double mean_phase = activation_sum > 0.0L ? phase_match / activation_sum : 0.0L;
        const long double score = 0.60L * mean_phase + 0.40L * std::log1p(static_cast<long double>(count));
        if (score > 0.12L) {
            scored.push_back({token, score});
        }
    }

    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });

    std::size_t emitted = 0;
    std::string previous;
    for (const auto& item : scored) {
        if (emitted >= 32U) {
            break;
        }
        if (previous == item.token) {
            continue;
        }
        if (emitted > 0) {
            out << ' ';
        }
        out << item.token;
        previous = item.token;
        ++emitted;
    }
    if (emitted == 0) {
        return std::string(prompt) + ".";
    }
    out << ".";
    return out.str();
}

inline PhaseLanguageResult generate_phase_language(IuttEnsemble& ensemble,
                                                    const CodeTokenMemory& memory,
                                                    SemanticFieldMemory& semantic_memory,
                                                    std::string_view prompt,
                                                    PhaseLanguageConfig config = {}) {
    PhaseLanguageResult result;
    if (config.mode == PhaseLanguageMode::Natural) {
        config.mode = detect_phase_language_mode(prompt);
    }
    const auto iutt = ensemble.resonate(prompt, 0.011L);
    auto field = make_field_state(ensemble.main_cloud().active_handles(std::min<std::size_t>(config.field_context, 512)),
                                  prompt,
                                  std::min<std::size_t>(config.field_context, 512));
    const auto memory_hits = semantic_memory.resonate(prompt, field, 1);
    const long double memory_resonance = memory_hits.empty() ? 0.0L : memory_hits.front().score;
    const long double syntax_seed = config.mode == PhaseLanguageMode::Python ? 0.0L : 0.0L;
    const auto attractor = run_variational_attractor(field,
                                                      std::max<std::size_t>(1, config.attractor_steps),
                                                      memory_resonance,
                                                      syntax_seed);
    result.energy = attractor.final_energy;

    if (config.mode == PhaseLanguageMode::Python) {
        result.output = phase_python_output(memory, field, prompt);
    } else {
        result.output = phase_decode_field_into_text(field, memory, prompt);
    }
    result.tokens = tokenize_code(result.output);
    const auto syntax_energy = config.mode == PhaseLanguageMode::Python
                                   ? python_phase_constraint_energy(result.output)
                                   : 0.0L;
    result.energy = evaluate_variational_energy(field, memory_resonance, syntax_energy);
    result.accepted = phase_language_accepts(result.output, config.mode) &&
                      (!config.reject_unstable_output || result.energy.total_free_energy < 1.2L);
    if (!result.accepted) {
        ++result.rejected_candidates;
        if (config.mode == PhaseLanguageMode::Python) {
            result.output = "def generated(value):\n    return value\n";
            result.tokens = tokenize_code(result.output);
            result.energy = evaluate_variational_energy(field, memory_resonance,
                                                         python_phase_constraint_energy(result.output));
            result.accepted = phase_language_accepts(result.output, config.mode);
        }
    }
    semantic_memory.imprint(prompt, field, result.energy, result.output, result.accepted);
    (void)iutt;
    return result;
}

} // namespace dzeta
