#pragma once

#include "broca.h"
#include "information.h"
#include "sat.h"
#include "task_state.h"
#include "world_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

enum class TaskKind {
    Unknown,
    Arithmetic,
    Planning,
    WorldQuery,
    Causal,
    FewShot,
    Transfer,
    LearnedOperator,
    Landscape,
    Resonance,
    Generated,
};

struct PlanStep {
    std::string action;
    std::string reason;
};

struct ExecutiveConfig {
    bool agi_core = false;
    bool learn_online = false;
    bool explain_trace = false;
    std::size_t max_thoughts = 8;
    std::size_t verify_candidates = 6;
    std::size_t broca_tokens = 32;
};

struct CandidateSolution {
    TaskKind kind = TaskKind::Unknown;
    std::string answer;
    std::vector<PlanStep> plan;
    std::string explanation;
    std::vector<std::string> trace;
    std::vector<std::string> transferred_from_domains;
    std::vector<CausalEdge> causal_edges_used;
    long double confidence = 0.0L;
    long double novelty = 0.0L;
    long double verification_score = 0.0L;
};

struct ExecutiveResult {
    TaskKind kind = TaskKind::Unknown;
    std::string answer;
    std::vector<PlanStep> plan;
    std::string explanation;
    std::vector<std::string> trace;
    std::vector<std::string> transferred_from_domains;
    std::vector<CausalEdge> causal_edges_used;
    std::vector<std::string> working_memory;
    long double confidence = 0.0L;
    long double novelty = 0.0L;
    long double verification_score = 0.0L;
    std::size_t candidates_considered = 0;
};

inline std::string lower_copy(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const unsigned char ch : text) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

inline std::string trim_copy(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return std::string{text.substr(begin, end - begin)};
}

inline bool contains_text(std::string_view haystack, std::string_view needle) {
    return lower_copy(haystack).find(lower_copy(needle)) != std::string::npos;
}

inline std::vector<std::string> tokenize_words(std::string_view text) {
    std::vector<std::string> words;
    std::string current;
    for (const unsigned char ch : text) {
        if (std::isalnum(ch) != 0 || ch == '_') {
            current.push_back(static_cast<char>(std::tolower(ch)));
        } else if (!current.empty()) {
            words.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

class ArithmeticParser {
public:
    explicit ArithmeticParser(std::string expression) : expression_(std::move(expression)) {}

    std::optional<long double> parse() {
        position_ = 0;
        const auto value = parse_expression();
        skip_ws();
        if (!value || position_ != expression_.size()) {
            return std::nullopt;
        }
        return value;
    }

private:
    std::optional<long double> parse_expression() {
        auto value = parse_term();
        if (!value) {
            return std::nullopt;
        }
        for (;;) {
            skip_ws();
            if (match('+')) {
                auto rhs = parse_term();
                if (!rhs) {
                    return std::nullopt;
                }
                *value += *rhs;
            } else if (match('-')) {
                auto rhs = parse_term();
                if (!rhs) {
                    return std::nullopt;
                }
                *value -= *rhs;
            } else {
                return value;
            }
        }
    }

    std::optional<long double> parse_term() {
        auto value = parse_factor();
        if (!value) {
            return std::nullopt;
        }
        for (;;) {
            skip_ws();
            if (match('*')) {
                auto rhs = parse_factor();
                if (!rhs) {
                    return std::nullopt;
                }
                *value *= *rhs;
            } else if (match('/')) {
                auto rhs = parse_factor();
                if (!rhs || std::abs(*rhs) < 1.0e-18L) {
                    return std::nullopt;
                }
                *value /= *rhs;
            } else {
                return value;
            }
        }
    }

    std::optional<long double> parse_factor() {
        skip_ws();
        if (match('(')) {
            auto value = parse_expression();
            if (!value || !match(')')) {
                return std::nullopt;
            }
            return value;
        }
        return parse_number();
    }

    std::optional<long double> parse_number() {
        skip_ws();
        const auto begin = position_;
        if (position_ < expression_.size() && (expression_[position_] == '+' || expression_[position_] == '-')) {
            ++position_;
        }
        bool saw_digit = false;
        while (position_ < expression_.size()) {
            const unsigned char ch = static_cast<unsigned char>(expression_[position_]);
            if (std::isdigit(ch) != 0) {
                saw_digit = true;
                ++position_;
            } else if (expression_[position_] == '.') {
                ++position_;
            } else {
                break;
            }
        }
        if (!saw_digit) {
            position_ = begin;
            return std::nullopt;
        }
        return std::stold(expression_.substr(begin, position_ - begin));
    }

    void skip_ws() {
        while (position_ < expression_.size() &&
               std::isspace(static_cast<unsigned char>(expression_[position_])) != 0) {
            ++position_;
        }
    }

    bool match(char expected) {
        skip_ws();
        if (position_ < expression_.size() && expression_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    std::string expression_;
    std::size_t position_ = 0;
};

inline std::string extract_arithmetic_expression(std::string_view prompt) {
    std::string expression;
    bool saw_operator = false;
    bool saw_digit = false;
    for (const char ch : prompt) {
        if (std::isdigit(static_cast<unsigned char>(ch)) != 0 || ch == '.' || ch == ' ' ||
            ch == '(' || ch == ')' || ch == '+' || ch == '-' || ch == '*' || ch == '/') {
            expression.push_back(ch);
            if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
                saw_digit = true;
            }
            if (ch == '+' || ch == '-' || ch == '*' || ch == '/') {
                saw_operator = true;
            }
        }
    }
    return saw_digit && saw_operator ? trim_copy(expression) : std::string{};
}

inline std::string format_number(long double value) {
    std::ostringstream out;
    if (std::abs(value - std::round(value)) < 1.0e-12L) {
        out << static_cast<long long>(std::llround(value));
    } else {
        out << static_cast<double>(value);
    }
    return out.str();
}

inline std::string infer_domain(std::string_view prompt) {
    const auto lower = lower_copy(prompt);
    if (lower.find("python") != std::string::npos) {
        return "python";
    }
    if (lower.find("javascript") != std::string::npos || lower.find("typescript") != std::string::npos) {
        return "javascript";
    }
    if (lower.find("math") != std::string::npos || lower.find("solve") != std::string::npos) {
        return "math";
    }
    if (lower.find("plan") != std::string::npos || lower.find("build") != std::string::npos) {
        return "planning";
    }
    return "general";
}

inline std::size_t token_overlap(std::string_view left, std::string_view right) {
    const auto a = tokenize_words(left);
    const auto b = tokenize_words(right);
    std::size_t count = 0;
    for (const auto& word : a) {
        if (word.size() < 3) {
            continue;
        }
        if (std::find(b.begin(), b.end(), word) != b.end()) {
            ++count;
        }
    }
    return count;
}

inline std::string task_kind_name(TaskKind kind) {
    switch (kind) {
    case TaskKind::Arithmetic:
        return "arithmetic";
    case TaskKind::Planning:
        return "planning";
    case TaskKind::WorldQuery:
        return "world_query";
    case TaskKind::Causal:
        return "causal";
    case TaskKind::FewShot:
        return "few_shot";
    case TaskKind::Transfer:
        return "transfer";
    case TaskKind::LearnedOperator:
        return "learned_operator";
    case TaskKind::Landscape:
        return "sat_landscape";
    case TaskKind::Resonance:
        return "math_resonance";
    case TaskKind::Generated:
        return "generated";
    case TaskKind::Unknown:
        return "unknown";
    }
    return "unknown";
}

class Executive {
public:
    explicit Executive(IuttEnsemble& ensemble, WorldModel* model = nullptr, ExecutiveConfig config = {})
        : ensemble_(ensemble), world_(model == nullptr ? &owned_world_ : model), config_(config) {}

    void observe_fact(std::string subject, std::string predicate, std::string object, std::string evidence) {
        world_->observe_fact(std::move(subject), std::move(predicate), std::move(object), std::move(evidence), 1.0L);
    }

    void observe_cause(std::string cause, std::string effect, std::string evidence, long double weight = 1.0L) {
        world_->observe_cause(std::move(cause), std::move(effect), std::move(evidence), weight);
    }

    void observe_example(std::string input, std::string output, std::string domain = "general") {
        world_->observe_example(std::move(input), std::move(output), std::move(domain));
    }

    ExecutiveResult solve(std::string_view prompt) {
        auto state = make_task_state(prompt);
        auto candidates = generate_candidates(state);
        if (candidates.empty()) {
            CandidateSolution generated;
            solve_generated(state, generated);
            candidates.push_back(std::move(generated));
        }
        verify_candidate_set(state, candidates);
        const auto selected = select_candidate(candidates);
        record_episodes(state, candidates, selected);
        return to_result(state, selected, candidates.size());
    }

private:
    using OperatorFn = bool (Executive::*)(const TaskState&, CandidateSolution&) const;

    struct ThoughtOperator {
        const char* name = "";
        TaskKind kind = TaskKind::Unknown;
        OperatorFn run = nullptr;
    };

    TaskState make_task_state(std::string_view prompt) {
        TaskState state;
        state.goal = trim_copy(prompt);
        if (state.goal.empty()) {
            state.goal = "empty task";
        }
        state.domain = infer_domain(state.goal);
        state.tokens = tokenize_words(state.goal);
        state.trace.push_back("received task");
        state.trace.push_back("domain=" + state.domain);
        state.resonance = ensemble_.resonate(state.goal, 0.017L);
        state.trace.push_back(state.resonance.accepted ? "resonance accepted" : "resonance below invariant threshold");
        state.working_memory.push_back("goal:" + state.goal);
        state.working_memory.push_back("domain:" + state.domain);
        state.working_memory.push_back("tokens:" + std::to_string(state.tokens.size()));
        state.working_memory.push_back("bridge_score:" + format_number(state.resonance.invariant.score));
        if (state.resonance.snapshot_a.max_clique_found > 0) {
            state.working_memory.push_back("max_clique:" + std::to_string(state.resonance.snapshot_a.max_clique_found));
        }
        return state;
    }

    std::vector<ThoughtOperator> operator_catalog() const {
        std::vector<ThoughtOperator> ops;
        if (config_.agi_core || config_.learn_online) {
            ops.push_back({"learned_operator", TaskKind::LearnedOperator, &Executive::solve_learned_operator});
        }
        ops.push_back({"arithmetic", TaskKind::Arithmetic, &Executive::solve_arithmetic});
        ops.push_back({"world_query", TaskKind::WorldQuery, &Executive::solve_world_query});
        ops.push_back({"causal", TaskKind::Causal, &Executive::solve_causal});
        ops.push_back({"transfer", TaskKind::Transfer, &Executive::solve_transfer});
        ops.push_back({"few_shot", TaskKind::FewShot, &Executive::solve_few_shot});
        ops.push_back({"planning", TaskKind::Planning, &Executive::solve_planning});
        if (config_.agi_core) {
            ops.push_back({"sat_landscape", TaskKind::Landscape, &Executive::solve_landscape});
            ops.push_back({"math_resonance", TaskKind::Resonance, &Executive::solve_resonance});
        }
        ops.push_back({"broca", TaskKind::Generated, &Executive::solve_generated});
        return ops;
    }

    std::vector<CandidateSolution> generate_candidates(const TaskState& state) const {
        std::vector<CandidateSolution> candidates;
        const auto thought_limit = std::max<std::size_t>(1, config_.max_thoughts);
        for (const auto& op : operator_catalog()) {
            if (candidates.size() >= thought_limit) {
                break;
            }
            CandidateSolution candidate;
            candidate.kind = op.kind;
            if ((this->*op.run)(state, candidate)) {
                candidate.trace.push_back(std::string("operator considered: ") + op.name);
                candidates.push_back(std::move(candidate));
            }
        }
        return candidates;
    }

    void verify_candidate_set(const TaskState& state, std::vector<CandidateSolution>& candidates) const {
        std::stable_sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
            return left.confidence > right.confidence;
        });
        const auto verify_limit = std::max<std::size_t>(1, config_.verify_candidates);
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            candidates[i].novelty = novelty_score(candidates[i].answer);
            if (i < verify_limit) {
                candidates[i].verification_score = verify_candidate(state, candidates[i]);
            } else {
                candidates[i].verification_score = std::clamp(0.35L * candidates[i].confidence +
                                                              0.15L * candidates[i].novelty,
                                                              0.0L, 1.0L);
                candidates[i].trace.push_back("not deeply verified: outside verify candidate window");
            }
        }
    }

    CandidateSolution select_candidate(const std::vector<CandidateSolution>& candidates) const {
        return *std::max_element(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
            const long double left_score = 0.78L * left.verification_score +
                                           0.12L * left.novelty +
                                           0.10L * left.confidence;
            const long double right_score = 0.78L * right.verification_score +
                                            0.12L * right.novelty +
                                            0.10L * right.confidence;
            return left_score < right_score;
        });
    }

    ExecutiveResult to_result(const TaskState& state,
                              CandidateSolution selected,
                              std::size_t candidates_considered) const {
        ExecutiveResult result;
        result.kind = selected.kind;
        result.answer = std::move(selected.answer);
        result.plan = std::move(selected.plan);
        result.explanation = std::move(selected.explanation);
        result.trace = state.trace;
        result.trace.insert(result.trace.end(), selected.trace.begin(), selected.trace.end());
        result.trace.push_back("selected operator: " + task_kind_name(result.kind));
        result.transferred_from_domains = std::move(selected.transferred_from_domains);
        result.causal_edges_used = std::move(selected.causal_edges_used);
        result.working_memory = state.working_memory;
        result.confidence = std::clamp(0.65L * selected.verification_score + 0.35L * selected.confidence, 0.0L, 1.0L);
        result.novelty = selected.novelty;
        result.verification_score = selected.verification_score;
        result.candidates_considered = candidates_considered;
        return result;
    }

    void record_episodes(const TaskState& state,
                         const std::vector<CandidateSolution>& candidates,
                         const CandidateSolution& selected) {
        if (!config_.agi_core && !config_.learn_online) {
            return;
        }
        for (const auto& candidate : candidates) {
            const bool success = candidate.kind == selected.kind &&
                                 candidate.answer == selected.answer &&
                                 candidate.verification_score >= 0.5L;
            world_->observe_episode(state.goal,
                                    candidate.answer,
                                    candidate.verification_score,
                                    success,
                                    task_kind_name(candidate.kind));
        }
    }

    long double novelty_score(std::string_view answer) const {
        if (answer.empty()) {
            return 0.0L;
        }
        if (world_->episodes().empty()) {
            return 1.0L;
        }
        long double nearest = 0.0L;
        for (const auto& episode : world_->episodes()) {
            const auto overlap = token_overlap(answer, episode.answer);
            const auto denom = std::max<std::size_t>(1, tokenize_words(answer).size());
            nearest = std::max(nearest, static_cast<long double>(overlap) / static_cast<long double>(denom));
        }
        return std::clamp(1.0L - nearest, 0.0L, 1.0L);
    }

    long double verify_candidate(const TaskState& state, CandidateSolution& candidate) const {
        long double score = 0.15L * candidate.confidence +
                            0.10L * state.resonance.invariant.score;
        if (!candidate.answer.empty()) {
            score += 0.08L;
        }
        if (!candidate.plan.empty()) {
            score += std::min<long double>(0.12L, 0.03L * static_cast<long double>(candidate.plan.size()));
        }
        if (!candidate.explanation.empty()) {
            score += 0.05L;
        }
        if (token_overlap(state.goal, candidate.answer) > 0) {
            score += 0.06L;
        }

        switch (candidate.kind) {
        case TaskKind::Arithmetic:
            score += 0.70L;
            break;
        case TaskKind::WorldQuery:
            score += verify_world_answer(state, candidate);
            break;
        case TaskKind::Causal:
            score += candidate.causal_edges_used.empty()
                         ? 0.15L
                         : 0.45L + 0.25L * candidate.causal_edges_used.front().weight;
            break;
        case TaskKind::LearnedOperator:
            score += verify_learned_operator(state, candidate);
            break;
        case TaskKind::FewShot:
            score += 0.45L;
            break;
        case TaskKind::Transfer:
            score += candidate.transferred_from_domains.empty() ? 0.20L : 0.45L;
            break;
        case TaskKind::Planning:
            score += candidate.plan.size() >= 3 ? 0.35L : 0.15L;
            break;
        case TaskKind::Landscape:
            score += 0.30L + 0.10L * std::min<long double>(1.0L, state.resonance.snapshot_a.mutual_information);
            break;
        case TaskKind::Resonance:
            score += 0.10L + 0.15L * state.resonance.invariant.score;
            break;
        case TaskKind::Generated:
            score += 0.12L;
            break;
        case TaskKind::Unknown:
            break;
        }

        const auto complexity = kolmogorov_upper_bound(candidate.answer);
        score -= 0.04L * complexity.normalized_complexity;
        candidate.trace.push_back("verification=" + format_number(score));
        return std::clamp(score, 0.0L, 1.0L);
    }

    long double verify_world_answer(const TaskState& state, const CandidateSolution& candidate) const {
        long double best = 0.0L;
        for (const auto& fact : world_->facts()) {
            const bool prompt_match = contains_text(state.goal, fact.object) ||
                                      contains_text(state.goal, fact.subject);
            const bool answer_match = contains_text(candidate.answer, fact.subject) &&
                                      contains_text(candidate.answer, fact.object);
            if (prompt_match && answer_match) {
                best = std::max(best, 0.35L + 0.45L * fact.confidence);
            }
        }
        return best;
    }

    long double verify_learned_operator(const TaskState& state, const CandidateSolution& candidate) const {
        long double best = 0.0L;
        for (const auto& op : world_->learned_operators()) {
            if (!learned_domain_compatible(op.domain, state.domain)) {
                continue;
            }
            const auto applied = op.apply(state.goal);
            if (applied && *applied == candidate.answer) {
                best = std::max(best, 0.35L + 0.45L * op.confidence);
            }
        }
        return best;
    }

    bool solve_arithmetic(const TaskState& state, CandidateSolution& result) const {
        const auto expression = extract_arithmetic_expression(state.goal);
        if (expression.empty()) {
            return false;
        }
        ArithmeticParser parser(expression);
        const auto value = parser.parse();
        if (!value) {
            return false;
        }
        result.kind = TaskKind::Arithmetic;
        result.plan = {
            {"extract arithmetic expression", expression},
            {"parse expression", "recursive descent with operator precedence"},
            {"evaluate expression", "produce numeric result"},
        };
        result.answer = format_number(*value);
        result.explanation = "I extracted `" + expression + "`, applied operator precedence, and evaluated it.";
        result.confidence = 0.96L;
        result.trace.push_back("arithmetic solver produced exact numeric candidate");
        return true;
    }

    bool solve_world_query(const TaskState& state, CandidateSolution& result) const {
        const auto lower = lower_copy(state.goal);
        const Fact* best = nullptr;
        long double best_score = 0.0L;
        for (const auto& fact : world_->facts()) {
            long double match = 0.0L;
            if (fact.predicate == "capital_of" && lower.find("capital") != std::string::npos &&
                lower.find(lower_copy(fact.object)) != std::string::npos) {
                match = 1.0L;
            } else if ((contains_text(state.goal, fact.subject) || contains_text(state.goal, fact.object)) &&
                       contains_text(state.goal, fact.predicate)) {
                match = 0.85L;
            }
            match *= fact.confidence;
            if (match > best_score) {
                best_score = match;
                best = &fact;
            }
        }
        if (best == nullptr) {
            return false;
        }
        result.kind = TaskKind::WorldQuery;
        result.plan = {
            {"identify queried relation", best->predicate},
            {"search persistent world model", best->object},
            {"return fact with evidence", best->evidence},
        };
        if (best->predicate == "capital_of") {
            result.answer = best->subject + " is the capital of " + best->object + ".";
        } else {
            result.answer = best->subject + " " + best->predicate + " " + best->object + ".";
        }
        result.explanation = "The answer comes from persistent world-model evidence: " + best->evidence + ".";
        result.confidence = std::clamp(0.55L + 0.40L * best_score, 0.0L, 1.0L);
        result.trace.push_back("world model fact matched");
        return true;
    }

    bool solve_causal(const TaskState& state, CandidateSolution& result) const {
        const auto lower = lower_copy(state.goal);
        if (lower.find("why") == std::string::npos && lower.find("cause") == std::string::npos) {
            return false;
        }
        const CausalEdge* best = nullptr;
        long double best_score = 0.0L;
        for (const auto& edge : world_->causes()) {
            long double match = 0.0L;
            if (lower.find(lower_copy(edge.effect)) != std::string::npos) {
                match = 1.0L;
            } else if (token_overlap(state.goal, edge.effect) > 0) {
                match = 0.55L;
            }
            match *= edge.weight;
            if (match > best_score) {
                best = &edge;
                best_score = match;
            }
        }
        if (best == nullptr) {
            return false;
        }
        result.kind = TaskKind::Causal;
        result.causal_edges_used.push_back(*best);
        result.plan = {
            {"identify effect", best->effect},
            {"search causal graph", best->cause + " -> " + best->effect},
            {"explain causal link", best->evidence},
        };
        result.answer = best->effect + " is likely caused by " + best->cause + ".";
        result.explanation = "Causal edge used: " + best->cause + " -> " + best->effect +
                             " because " + best->evidence + ".";
        result.confidence = std::clamp(0.50L + 0.40L * best_score, 0.0L, 1.0L);
        result.trace.push_back("causal graph edge matched");
        return true;
    }

    bool solve_transfer(const TaskState& state, CandidateSolution& result) const {
        if (state.domain == "general") {
            return false;
        }
        const FewShotExample* best = nullptr;
        std::size_t best_overlap = 0;
        for (const auto& example : world_->examples()) {
            if (example.domain == state.domain) {
                continue;
            }
            const auto overlap = token_overlap(state.goal, example.input);
            if (overlap > best_overlap) {
                best_overlap = overlap;
                best = &example;
            }
        }
        if (best == nullptr || best_overlap == 0) {
            return false;
        }
        result.kind = TaskKind::Transfer;
        result.transferred_from_domains.push_back(best->domain);
        result.plan = {
            {"infer target domain", state.domain},
            {"find related prior skill", best->domain + ": " + best->input},
            {"project reusable structure", best->output},
        };
        result.answer = "Transfer from " + best->domain + ": " + best->output;
        result.explanation = "I reused a prior example from `" + best->domain +
                             "` because it shares task tokens with the new domain.";
        result.confidence = std::clamp(0.45L + 0.12L * static_cast<long double>(best_overlap), 0.0L, 0.9L);
        result.trace.push_back("cross-domain transfer selected");
        return true;
    }

    bool solve_few_shot(const TaskState& state, CandidateSolution& result) const {
        if (world_->examples().size() < 2) {
            return false;
        }
        std::optional<LearnedOperator> pattern;
        for (const auto& example : world_->examples()) {
            auto inferred = infer_learned_operator(example.input, example.output, example.domain);
            if (!inferred) {
                return false;
            }
            if (!pattern) {
                pattern = *inferred;
            } else if (pattern->kind != inferred->kind || pattern->parameter != inferred->parameter) {
                return false;
            }
        }
        const auto transformed = pattern->apply(state.goal);
        if (!transformed) {
            return false;
        }
        result.kind = TaskKind::FewShot;
        result.plan = {
            {"inspect examples", "examples share transform `" + pattern->name() + "`"},
            {"infer transformation", pattern->name()},
            {"apply transformation", state.goal},
        };
        result.answer = *transformed;
        result.explanation = "The few-shot examples define `" + pattern->name() + "`, so I applied it.";
        result.confidence = 0.78L;
        result.trace.push_back("few-shot transform inferred");
        return true;
    }

    bool solve_learned_operator(const TaskState& state, CandidateSolution& result) const {
        const auto transformed = world_->apply_learned_operator(state.goal, state.domain);
        if (!transformed || *transformed == state.goal) {
            return false;
        }
        result.kind = TaskKind::LearnedOperator;
        result.plan = {
            {"load learned operator", "persistent one/few-shot operator memory"},
            {"apply operator", state.goal},
            {"verify output", "compare with operator rule and bridge state"},
        };
        result.answer = *transformed;
        result.explanation = "A persistent learned operator transformed the input.";
        result.confidence = 0.86L;
        result.trace.push_back("persistent learned operator applied");
        return true;
    }

    bool solve_planning(const TaskState& state, CandidateSolution& result) const {
        const auto lower = lower_copy(state.goal);
        if (lower.find("plan") == std::string::npos && lower.find("build") == std::string::npos &&
            !config_.agi_core) {
            return false;
        }
        result.kind = TaskKind::Planning;
        result.plan = {
            {"define target", state.goal},
            {"generate candidate actions", "use task state, memory, and available operators"},
            {"verify candidates", "score each candidate against evidence and DZETA resonance"},
            {"adapt memory", "store outcome as an episode for future transfer"},
        };
        std::ostringstream answer;
        answer << "Plan:\n";
        for (std::size_t i = 0; i < result.plan.size(); ++i) {
            answer << (i + 1) << ". " << result.plan[i].action << " - " << result.plan[i].reason << "\n";
        }
        result.answer = answer.str();
        result.explanation = "I selected planning because the task asks for a sequence of actions or AGI core requested deliberation.";
        result.confidence = 0.66L;
        result.trace.push_back("planner produced candidate");
        return true;
    }

    bool solve_landscape(const TaskState& state, CandidateSolution& result) const {
        const auto problem = sat_from_query(state.goal);
        const auto walkers = std::max<std::size_t>(4, std::min<std::size_t>(8, state.tokens.size() + 2));
        const auto spectrum = complexity_spectrum(problem, walkers, 32, state.resonance.seed);
        result.kind = TaskKind::Landscape;
        result.plan = {
            {"encode task as SAT landscape", std::to_string(problem.variable_count) + " variables"},
            {"walk landscape", std::to_string(spectrum.walk_times.size()) + " walkers"},
            {"derive depth signal", "temperature_hint=" + format_number(spectrum.temperature_hint)},
        };
        result.answer = "SAT landscape: variables=" + std::to_string(problem.variable_count) +
                        " clauses=" + std::to_string(problem.clauses.size()) +
                        " temperature_hint=" + format_number(spectrum.temperature_hint) + ".";
        result.explanation = "This candidate does not solve SAT; it uses the landscape as a difficulty/depth signal.";
        result.confidence = 0.44L;
        result.trace.push_back("complexity landscape candidate generated");
        return true;
    }

    bool solve_resonance(const TaskState& state, CandidateSolution& result) const {
        result.kind = TaskKind::Resonance;
        result.plan = {
            {"resonate ensemble", "A/B/C theaters already evaluated"},
            {"read invariant", state.resonance.invariant.reason},
            {"expose math trace", "bridge_score=" + format_number(state.resonance.invariant.score)},
        };
        result.answer = std::string("Math resonance ") +
                        (state.resonance.accepted ? "accepted" : "rejected") +
                        " with score=" + format_number(state.resonance.invariant.score) + ".";
        result.explanation = "This candidate exposes the current DZETA invariant state instead of inventing a direct answer.";
        result.confidence = std::clamp(0.18L + 0.25L * state.resonance.invariant.score, 0.0L, 0.65L);
        result.trace.push_back("math resonance candidate generated");
        return true;
    }

    bool solve_generated(const TaskState& state, CandidateSolution& result) const {
        result.kind = TaskKind::Generated;
        result.plan = {
            {"resonate prompt", "activate DZETA ensemble"},
            {"route to Broca", "generate a coherent response from task state"},
            {"return output", "fallback when no verified specialized candidate dominates"},
        };
        Broca broca;
        result.answer = broca.generate(ensemble_, state, config_.broca_tokens);
        result.explanation = "Broca generated from the task state and recursive DZETA feedback.";
        result.confidence = 0.30L;
        result.trace.push_back("broca generation candidate produced");
        return !result.answer.empty();
    }

    IuttEnsemble& ensemble_;
    WorldModel owned_world_;
    WorldModel* world_ = nullptr;
    ExecutiveConfig config_;
};

} // namespace dzeta
