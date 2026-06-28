#pragma once

#include "code_memory.h"
#include "iutt.h"
#include "task_state.h"
#include "variational_core.h"
#include "field_state.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <initializer_list>

namespace dzeta {

enum class BrocaMode {
    Natural,
    Code,
};

struct BrocaState {
    std::vector<std::string> context_tokens;
    std::deque<std::string> recent;
    int parens = 0;
    int brackets = 0;
    int braces = 0;
    bool produced_statement = false;
    FieldState field;

    void seed(std::string_view prompt) {
        context_tokens.clear();
        recent.clear();
        parens = 0;
        brackets = 0;
        braces = 0;
        produced_statement = false;
        for (const auto& token : tokenize_code(prompt)) {
            apply(token);
        }
    }

    void apply(const std::string& token) {
        context_tokens.push_back(token);
        if (token == "(") {
            ++parens;
        } else if (token == ")" && parens > 0) {
            --parens;
        } else if (token == "[") {
            ++brackets;
        } else if (token == "]" && brackets > 0) {
            --brackets;
        } else if (token == "{") {
            ++braces;
        } else if (token == "}" && braces > 0) {
            --braces;
        } else if (token == "\n") {
            produced_statement = true;
        }

        recent.push_back(token);
        while (recent.size() > 12) {
            recent.pop_front();
        }
    }

    bool balanced() const noexcept {
        return parens == 0 && brackets == 0 && braces == 0;
    }

    std::string previous() const {
        return context_tokens.empty() ? std::string{} : context_tokens.back();
    }

    std::string before_previous() const {
        return context_tokens.size() < 2 ? std::string{} : context_tokens[context_tokens.size() - 2];
    }
};

struct BrocaConfig {
    std::size_t candidate_limit = 96;
    std::size_t recent_penalty_window = 12;
    long double frequency_weight = 0.18L;
    long double resonance_weight = 0.50L;
    long double syntax_weight = 0.24L;
    long double context_weight = 0.08L;
};

inline long double token_field_energy(const std::string& token, const FieldState& field) {
    if (field.empty()) {
        return 0.5L;
    }
    const auto hash = stable_hash(token);
    const std::size_t idx = static_cast<std::size_t>(hash % field.size());
    const long double phase_align = std::cos(field.phases[idx] + field.theta[idx]);
    const long double activation = field.activations[idx];
    const long double padic_gap = field.padic_coordinates[idx];
    return 1.0L - 0.5L * (phase_align + activation) / (1.0L + padic_gap);
}

inline void apply_field_attention(FieldState& field, const std::string& token) {
    if (field.empty()) {
        return;
    }
    const auto hash = stable_hash(token);
    for (std::size_t i = 0; i < field.size(); ++i) {
        const long double resonance = std::cos(static_cast<long double>(hash ^ field.primes[i]) * 0.001L);
        field.activations[i] = std::clamp(field.activations[i] + 0.04L * (resonance - 0.3L), 0.0L, 1.0L);
        field.semantic_charge[i] = 0.995L * field.semantic_charge[i] + 0.005L * resonance;
    }
}

inline long double token_to_field_coherence(const std::string& token, const FieldState& field) {
    if (field.empty()) {
        return 0.0L;
    }
    const auto hash = stable_hash(token);
    long double sum = 0.0L;
    for (std::size_t i = 0; i < std::min<std::size_t>(field.size(), 64U); ++i) {
        const long double phase_diff = wrap_phase(field.phases[i] - static_cast<long double>(hash % 1000) * 0.00628L);
        sum += std::cos(phase_diff) * field.activations[i];
    }
    return sum / static_cast<long double>(std::min<std::size_t>(field.size(), 64U));
}

class Broca {
public:
    explicit Broca(const CodeTokenMemory* lexicon = nullptr, BrocaConfig config = {})
        : lexicon_(lexicon), config_(config) {}

    std::string generate(IuttEnsemble& ensemble, const TaskState& task, std::size_t max_tokens) const {
        std::string prompt = task.goal;
        for (const auto& item : task.working_memory) {
            prompt += " ";
            prompt += item;
        }
        return generate(ensemble, prompt, max_tokens);
    }

    std::string generate(IuttEnsemble& ensemble, std::string_view prompt, std::size_t max_tokens) const {
        const auto mode = detect_mode(prompt);
        BrocaState state;
        state.seed(prompt);

        auto active = ensemble.main_cloud().active_handles(256);
        state.field = make_field_state(active, prompt, 256);
        if (!state.field.empty()) {
            auto attractor = run_variational_attractor(state.field, 8, 0.0L, 0.0L);
            (void)attractor;
        }

        std::vector<std::string> emitted;
        emitted.reserve(max_tokens);
        std::string recursive_context(prompt);

        for (std::size_t step = 0; step < max_tokens; ++step) {
            const auto response = ensemble.resonate(recursive_context, 0.007L * static_cast<long double>(step + 1));
            const auto candidates = build_candidates(prompt, state, mode);
            if (candidates.empty()) {
                break;
            }

            const auto token = select_token(candidates, state, response, recursive_context, step, mode);
            if (token.empty()) {
                break;
            }
            emitted.push_back(token);
            state.apply(token);
            apply_field_attention(state.field, token);
            recursive_context.push_back(' ');
            recursive_context += token;

            if (step % 3U == 0U && !state.field.empty()) {
                variational_step(state.field, 0.04L);
            }

            if (should_stop(state, emitted, response, step)) {
                break;
            }
        }

        auto full = tokenize_code(prompt);
        full.insert(full.end(), emitted.begin(), emitted.end());
        return mode == BrocaMode::Code ? format_code_tokens(full) : format_natural_tokens(full);
    }

private:
    std::vector<std::pair<std::string, std::size_t>> build_candidates(std::string_view prompt,
                                                                        const BrocaState& state,
                                                                        BrocaMode mode) const {
        std::unordered_map<std::string, std::size_t> merged;

        if (lexicon_ != nullptr && lexicon_->token_count() != 0) {
            const auto lexicon_take = mode == BrocaMode::Natural ? config_.candidate_limit * 8 : config_.candidate_limit;
            for (const auto& item : lexicon_->ranked_tokens(lexicon_take)) {
                if (mode == BrocaMode::Natural &&
                    (is_code_only_token(item.first) || is_natural_noise_token(item.first))) {
                    continue;
                }
                merged[item.first] += item.second;
            }
        }

        for (const auto& token : tokenize_code(prompt, 256)) {
            if (token != "\n") {
                merged[token] += 4;
            }
        }

        if (mode == BrocaMode::Code) {
            add_structural_candidates(state, merged);
        }

        std::vector<std::pair<std::string, std::size_t>> candidates(merged.begin(), merged.end());
        for (auto& candidate : candidates) {
            const long double field_coherence = token_to_field_coherence(candidate.first, state.field);
            candidate.second += static_cast<std::size_t>(std::max(0.0L, field_coherence * 8.0L));
        }

        std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
            if (left.second == right.second) {
                return left.first < right.first;
            }
            return left.second > right.second;
        });
        if (candidates.size() > config_.candidate_limit) {
            candidates.resize(config_.candidate_limit);
        }
        return candidates;
    }

    static BrocaMode detect_mode(std::string_view prompt) {
        const auto tokens = tokenize_code(prompt, 128);
        for (const auto& token : tokens) {
            if (token == "def" || token == "class" || token == "return" || token == "import" ||
                token == "lambda" || token == "{" || token == "}" || token == "->" || token == ":=") {
                return BrocaMode::Code;
            }
        }
        return BrocaMode::Natural;
    }

    static void add_structural_candidates(const BrocaState& state,
                                          std::unordered_map<std::string, std::size_t>& merged) {
        const auto previous = state.previous();
        const auto before = state.before_previous();

        if (previous == "def" || previous == "class" || previous == "function" || previous == "fn") {
            merged["generated"] += 10;
            merged["answer"] += 4;
        }
        if ((before == "def" || before == "function" || before == "fn") && is_identifier_token(previous)) {
            merged["("] += 16;
        }
        if (previous == "(") {
            merged["value"] += 8;
            merged["self"] += 6;
            merged[")"] += 4;
        }
        if (state.parens > 0 && is_identifier_token(previous)) {
            merged[")"] += 14;
            merged[","] += 5;
        }
        if (previous == ")" && state.parens == 0) {
            merged[":"] += 16;
        }
        if (previous == ":") {
            merged["\n"] += 16;
        }
        if (previous == "\n") {
            merged["return"] += 12;
            merged["if"] += 4;
        }
        if (previous == "return") {
            merged["value"] += 10;
            merged["None"] += 3;
        }
        if (!state.balanced()) {
            if (state.parens > 0) {
                merged[")"] += 10;
            }
            if (state.brackets > 0) {
                merged["]"] += 10;
            }
            if (state.braces > 0) {
                merged["}"] += 10;
            }
        }
    }

    std::string select_token(const std::vector<std::pair<std::string, std::size_t>>& candidates,
                             const BrocaState& state,
                             const IuttResult& response,
                             std::string_view recursive_context,
                             std::size_t step,
                             BrocaMode mode) const {
        long double best_score = -1.0e100L;
        std::string best = candidates.front().first;

        for (const auto& candidate : candidates) {
            const auto& token = candidate.first;

            const long double field_energy = token_field_energy(token, state.field);
            const long double field_coherence = token_to_field_coherence(token, state.field);
            const long double amplitude = state.field.empty() ? 0.5L :
                0.5L * (1.0L - field_energy) + 0.5L * field_coherence;

            const auto hash = stable_hash(token) ^ response.seed ^ stable_hash(recursive_context) ^
                              (step * 0x9e3779b97f4a7c15ULL);
            const long double resonance = static_cast<long double>(hash % 100'000ULL) / 100'000.0L;
            const long double frequency = std::log1p(static_cast<long double>(candidate.second));
            const long double syntax = mode == BrocaMode::Code ? syntax_score(state, token)
                                                                : natural_score(state, token);

            const long double amplitude_score = 0.35L * amplitude + 0.15L * std::abs(amplitude - 0.5L);
            const long double score = 0.25L * amplitude_score +
                                      0.20L * frequency +
                                      0.20L * resonance +
                                      0.20L * (resonance + response.invariant.score) +
                                      0.20L * syntax +
                                      0.15L * field_coherence;
            if (score > best_score) {
                best_score = score;
                best = token;
            }
        }
        return best;
    }

    static std::string best_present(const std::vector<std::pair<std::string, std::size_t>>& candidates,
                                    std::initializer_list<std::string_view> wanted,
                                    std::string fallback) {
        for (const auto wanted_token : wanted) {
            const auto found = std::find_if(candidates.begin(), candidates.end(), [&](const auto& candidate) {
                return candidate.first == wanted_token;
            });
            if (found != candidates.end()) {
                return found->first;
            }
        }
        return fallback;
    }

    static long double syntax_score(const BrocaState& state, const std::string& token) {
        long double score = 0.0L;
        const auto previous = state.previous();
        const auto before = state.before_previous();

        const auto recent_count = std::count(state.recent.begin(), state.recent.end(), token);
        score -= 2.0L * static_cast<long double>(recent_count);
        if (token == previous) {
            score -= 8.0L;
        }
        if (token == "\"...\"") {
            score -= 2.0L + 3.0L * static_cast<long double>(recent_count);
        }
        if ((token == ")" && state.parens == 0) || (token == "]" && state.brackets == 0) ||
            (token == "}" && state.braces == 0)) {
            score -= 10.0L;
        }

        if ((previous == "def" || previous == "class") && is_identifier_token(token)) {
            score += 8.0L;
        }
        if ((before == "def" || before == "function" || before == "fn") &&
            is_identifier_token(previous) && token == "(") {
            score += 8.0L;
        }
        if (previous == "(" && (is_identifier_token(token) || token == ")")) {
            score += 4.0L;
        }
        if (state.parens > 0 && is_literal_token(token)) {
            score -= 6.0L;
        }
        if (previous == ")" && token == ":") {
            score += 8.0L;
        }
        if (previous == ":" && token == "\n") {
            score += 8.0L;
        }
        if (previous == "\n" && (token == "return" || token == "if" || token == "for" ||
                                  token == "while" || token == "try")) {
            score += 5.0L;
        }
        if (previous == "return" && (is_identifier_token(token) || is_literal_token(token))) {
            score += 5.0L;
        }
        if (state.balanced() && token == "\n") {
            score += 1.5L;
        }
        if (!state.balanced() && token == "\n") {
            score -= 4.0L;
        }
        return score;
    }

    static long double natural_score(const BrocaState& state, const std::string& token) {
        long double score = 0.0L;
        const auto previous = state.previous();
        const auto recent_count = std::count(state.recent.begin(), state.recent.end(), token);
        score -= 2.5L * static_cast<long double>(recent_count);
        if (token == previous) {
            score -= 10.0L;
        }
        if (is_code_only_token(token)) {
            score -= 25.0L;
        }
        if (token == ".") {
            score += state.context_tokens.size() > 10 ? 2.0L : -2.0L;
        } else if (is_identifier_token(token) && !is_reserved_word(token)) {
            score += 2.5L;
        }
        if (previous == "." && token == ".") {
            score -= 10.0L;
        }
        return score;
    }

    static long double context_score(const BrocaState& state, const std::string& token) {
        if (std::find(state.context_tokens.begin(), state.context_tokens.end(), token) != state.context_tokens.end()) {
            return 1.0L;
        }
        return 0.0L;
    }

    static bool should_stop(const BrocaState& state,
                            const std::vector<std::string>& emitted,
                            const IuttResult& response,
                            std::size_t step) {
        const std::size_t math_min_steps = std::max<std::size_t>(8, response.snapshot_a.guaranteed_clique_r * 2U);
        if (step < math_min_steps) {
            return false;
        }
        if (state.balanced() && state.produced_statement &&
            (emitted.back() == "\n" || emitted.back() == "}" || emitted.back() == ".")) {
            return true;
        }
        return response.accepted && response.snapshot_a.phase == KernerPhase::WideMovingJam;
    }

    static bool is_literal_token(const std::string& token) {
        if (token.empty()) {
            return false;
        }
        const unsigned char first = static_cast<unsigned char>(token.front());
        return std::isdigit(first) != 0 || token == "\"...\"" || token == "True" ||
               token == "False" || token == "None";
    }

    static bool is_code_only_token(const std::string& token) {
        static const std::vector<std::string> code_only{
            "\n", "def", "class", "return", "self", "None", "True", "False", "import",
            "from", "lambda", "try", "except", "finally", "yield", "pass", "raise",
            "(", ")", "[", "]", "{", "}", ":", "->", ":=", "=", "+=", "-=", "*=",
            "/=", "**", "//", "\"...\"",
        };
        return std::find(code_only.begin(), code_only.end(), token) != code_only.end();
    }

    static bool is_natural_noise_token(const std::string& token) {
        if (token.size() <= 1 && token != ".") {
            return true;
        }
        if (!token.empty() && !is_identifier_token(token) && token != ".") {
            return true;
        }
        std::string lower = token;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        static const std::vector<std::string> stopwords{
            "the", "of", "and", "to", "in", "a", "an", "was", "were", "that", "is",
            "are", "for", "with", "as", "by", "on", "at", "from", "this", "these",
            "those", "it", "its", "be", "been", "being", "or", "but", "we", "our",
            "they", "their", "which", "who", "when", "where", "than", "then", "there",
            "have", "has", "had", "not", "no", "all", "can", "may", "also", "using",
            "used", "use", "study", "data", "figure", "fig", "table", "result",
            "results", "analysis", "based", "between", "within", "through",
            "et", "al", "appendix", "supplementary", "supplemental", "author",
            "usepackage", "documentclass", "begin", "end",
        };
        return std::find(stopwords.begin(), stopwords.end(), lower) != stopwords.end();
    }

    static bool recent_contains(const BrocaState& state, std::string_view token) {
        return std::any_of(state.recent.begin(), state.recent.end(), [&](const std::string& item) {
            return item == token;
        });
    }

    static bool is_identifier_token(const std::string& token) {
        if (token.empty() || !is_identifier_start(static_cast<unsigned char>(token.front()))) {
            return false;
        }
        return std::all_of(token.begin() + 1, token.end(), [](char ch) {
            return is_identifier_body(static_cast<unsigned char>(ch));
        });
    }

    static bool is_reserved_word(const std::string& token) {
        static const std::vector<std::string> reserved{
            "False", "None", "True", "and", "as", "assert", "async", "await", "break",
            "class", "continue", "def", "del", "elif", "else", "except", "finally",
            "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
            "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
            "Any", "Optional", "Union", "bool", "bytes", "dict", "float", "int", "list",
            "object", "set", "str", "tuple", "type",
        };
        return std::find(reserved.begin(), reserved.end(), token) != reserved.end();
    }

    static std::string format_natural_tokens(const std::vector<std::string>& tokens) {
        std::ostringstream out;
        std::string previous;
        for (const auto& token : tokens) {
            if (token == "\n" || is_code_only_token(token)) {
                continue;
            }
            if (!previous.empty() && token != "." && previous != "." && token != "," && token != ";" && token != ")") {
                out << ' ';
            }
            out << token;
            previous = token;
        }
        return out.str();
    }

    const CodeTokenMemory* lexicon_ = nullptr;
    BrocaConfig config_;
};

} // namespace dzeta
