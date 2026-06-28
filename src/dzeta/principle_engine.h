#pragma once

#include "mentalese_core.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dzeta {

enum class PrincipleRelation {
    Cause,
    Sequence,
    PartOf,
    Definition,
    ConditionAction
};

struct PrincipleEdge {
    std::string source;
    PrincipleRelation relation = PrincipleRelation::Definition;
    std::string target;
    std::string evidence;
    long double strength = 0.0L;
};

struct PrincipleAnswer {
    std::string text;
    std::vector<PrincipleEdge> edges;
    long double confidence = 0.0L;
};

class PrincipleEngine {
public:
    void observe_text(std::string_view text) {
        std::string current;
        std::string active_process;
        std::size_t active_process_ttl = 0;
        for (const char ch : text) {
            if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') {
                const bool touched_process = observe_sentence(current, active_process);
                if (touched_process) {
                    active_process_ttl = 4;
                } else if (active_process_ttl != 0) {
                    --active_process_ttl;
                    if (active_process_ttl == 0) {
                        active_process.clear();
                    }
                }
                current.clear();
            } else {
                current.push_back(ch);
            }
        }
        (void)observe_sentence(current, active_process);
    }

    std::optional<PrincipleAnswer> answer(std::string_view query) const {
        const auto lower = normalize_phrase(query);
        const auto query_tokens = tokenize_query(lower);
        if (query_tokens.empty()) {
            return std::nullopt;
        }

        if (asks_process(query_tokens, lower)) {
            if (const auto process = best_process(query_tokens)) {
                return process_answer(*process);
            }
        }
        if (asks_why_or_cause(query_tokens, lower)) {
            if (const auto chain = causal_chain_answer(query_tokens)) {
                return chain;
            }
            if (const auto transitive = transitive_causal_chain(query_tokens)) {
                return transitive;
            }
        }
        if (asks_parts(query_tokens, lower)) {
            if (const auto parts = relation_answer(query_tokens, PrincipleRelation::PartOf, "parts")) {
                return parts;
            }
        }
        if (asks_definition(query_tokens, lower)) {
            if (const auto definition = relation_answer(query_tokens, PrincipleRelation::Definition, "definition")) {
                return definition;
            }
        }
        if (asks_contradictions(query_tokens)) {
            if (const auto contradictions = detect_contradictions(query_tokens)) {
                return contradictions;
            }
        }
        return std::nullopt;
    }

    bool has_contradiction(std::string_view a, std::string_view b) const {
        const auto norm_a = normalize_entity(std::string(a));
        const auto norm_b = normalize_entity(std::string(b));
        if (norm_a.empty() || norm_b.empty()) {
            return false;
        }
        for (const auto& edge : edges_) {
            if ((edge.relation == PrincipleRelation::Definition || edge.relation == PrincipleRelation::Cause || edge.relation == PrincipleRelation::Sequence) && edge.strength > 0.6L) {
                const bool a_implies_b = (edge.source == norm_a && edge.target == norm_b);
                const bool b_implies_a = (edge.source == norm_b && edge.target == norm_a);
                if (a_implies_b || b_implies_a) {
                    for (const auto& other : edges_) {
                        if ((other.relation == PrincipleRelation::Definition || other.relation == PrincipleRelation::Cause || other.relation == PrincipleRelation::Sequence) && other.strength > 0.6L) {
                            if (a_implies_b && other.source == norm_a && other.target != norm_b) {
                                return true;
                            }
                            if (b_implies_a && other.source == norm_b && other.target != norm_a) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }

    std::size_t edge_count() const noexcept {
        return edges_.size();
    }

    std::size_t process_count() const noexcept {
        return process_steps_.size();
    }

    const std::vector<PrincipleEdge>& edges() const noexcept {
        return edges_;
    }

    const std::unordered_map<std::string, std::vector<std::string>>& processes() const noexcept {
        return process_steps_;
    }

    void restore_edge(PrincipleEdge edge) {
        add_edge(std::move(edge));
    }

    void restore_process(std::string process, std::vector<std::string> steps) {
        process = normalize_entity(std::move(process));
        if (process.empty() || steps.empty()) {
            return;
        }
        auto& target = process_steps_[process];
        for (auto& step : steps) {
            step = normalize_entity(std::move(step));
            if (!step.empty() && std::find(target.begin(), target.end(), step) == target.end()) {
                target.push_back(std::move(step));
            }
        }
    }

    PrincipleAnswer answer_causal_why(const std::string& effect) const {
        const auto norm_effect = normalize_entity(effect);
        if (norm_effect.empty()) return {};
        std::vector<std::string> chain;
        std::unordered_set<std::string> visited;
        std::string current = norm_effect;
        for (std::size_t depth = 0; depth < 5U; ++depth) {
            if (!visited.insert(current).second) break;
            chain.push_back(current);
            const auto prev = std::find_if(edges_.begin(), edges_.end(), [&](const auto& e) {
                return (e.relation == PrincipleRelation::Cause ||
                        e.relation == PrincipleRelation::Sequence) &&
                       e.target == current && e.strength > 0.3L;
            });
            if (prev == edges_.end()) break;
            current = prev->source;
        }
        if (chain.size() < 2U) return {};
        std::reverse(chain.begin(), chain.end());
        PrincipleAnswer answer;
        std::ostringstream out;
        for (std::size_t i = 0; i < chain.size(); ++i) {
            if (i != 0) out << " -> ";
            out << chain[i];
            if (i != 0) {
                const auto cause_edge = std::find_if(edges_.begin(), edges_.end(), [&](const auto& e) {
                    return (e.relation == PrincipleRelation::Cause ||
                            e.relation == PrincipleRelation::Sequence) &&
                           e.source == chain[i - 1] && e.target == chain[i];
                });
                if (cause_edge != edges_.end()) answer.edges.push_back(*cause_edge);
            }
        }
        answer.text = "why " + chain.back() + "? " + out.str();
        answer.confidence = std::min(0.92L, 0.42L + 0.10L * static_cast<long double>(chain.size()));
        return answer;
    }

    std::vector<PrincipleAnswer> check_consistency() const {
        std::vector<PrincipleAnswer> inconsistencies;
        std::unordered_map<std::string, std::vector<const PrincipleEdge*>> source_map;
        for (const auto& edge : edges_) {
            if (edge.strength > 0.4L) {
                source_map[edge.source].push_back(&edge);
            }
        }
        for (const auto& [src, edges_from] : source_map) {
            for (std::size_t i = 0; i < edges_from.size(); ++i) {
                for (std::size_t j = i + 1; j < edges_from.size(); ++j) {
                    const auto& a = *edges_from[i];
                    const auto& b = *edges_from[j];
                    if (a.target == b.target) continue;
                    if (has_contradiction(a.target, b.target)) {
                        PrincipleAnswer ca;
                        ca.text = "inconsistency: " + src + " leads to both '" +
                                  a.target + "' and '" + b.target + "' which conflict";
                        ca.edges = {a, b};
                        ca.confidence = 0.65L;
                        inconsistencies.push_back(std::move(ca));
                    }
                }
            }
        }
        return inconsistencies;
    }

private:
    bool observe_sentence(std::string sentence,
                          std::string& active_process) {
        sentence = normalize_phrase(std::move(sentence));
        if (sentence.size() < 6U) {
            return false;
        }

        bool touched_process = false;
        if (const auto edge = parse_condition_action(sentence)) {
            add_edge(*edge);
        }
        if (const auto edge = parse_binary_relation(sentence,
                                                    {" causes ", " cause ", " creates ", " produce ", " produces ",
                                                     " leads to ", " lead to ", " results in ", " result in ",
                                                     " вызывает ", " приводить к ", " приводит к "},
                                                    PrincipleRelation::Cause)) {
            add_edge(*edge);
        }
        if (const auto edge = parse_binary_relation(sentence,
                                                    {" consists of ", " includes ", " contains ", " made of ",
                                                     " состоит из ", " включает ", " содержит "},
                                                    PrincipleRelation::PartOf)) {
            add_edge(*edge);
        }
        if (const auto edge = parse_binary_relation(sentence,
                                                    {" is a ", " is an ", " is the ", " is ", " это ", " является "},
                                                    PrincipleRelation::Definition)) {
            add_edge(*edge);
        }
        touched_process = observe_process_sentence(sentence, active_process);
        return touched_process;
    }

    bool observe_process_sentence(const std::string& sentence,
                                  std::string& active_process) {
        const auto begins = find_any(sentence, {" begins with ", " starts with ", " start with ", " начинается с "});
        if (begins.first != std::string::npos) {
            active_process = normalize_entity(sentence.substr(0, begins.first));
            const auto step = normalize_entity(sentence.substr(begins.first + begins.second.size()));
            if (!active_process.empty() && !step.empty()) {
                add_process_step(active_process, step, sentence);
                return true;
            }
            return false;
        }

        const auto after = find_any(sentence, {" after ", " then ", " next ", " затем ", " потом ", " после "});
        const auto leading_after = leading_sequence_marker(sentence);
        if (!active_process.empty() && (after.first != std::string::npos || !leading_after.empty())) {
            auto step = !leading_after.empty()
                            ? sentence.substr(leading_after.size())
                            : sentence.substr(after.first + after.second.size());
            if (const auto comma = step.find(','); comma != std::string::npos && comma + 1U < step.size()) {
                step = step.substr(comma + 1U);
            }
            step = remove_trailing_verbs(normalize_entity(std::move(step)));
            if (!step.empty()) {
                add_process_step(active_process, step, sentence);
                return true;
            }
        }
        return false;
    }

    static std::optional<PrincipleEdge> parse_condition_action(const std::string& sentence) {
        const auto if_pos = sentence.find("if ");
        const auto then_pos = sentence.find(" then ");
        if (if_pos == std::string::npos || then_pos == std::string::npos || then_pos <= if_pos + 3U) {
        return std::nullopt;
    }
        const auto condition = normalize_entity(sentence.substr(if_pos + 3U, then_pos - if_pos - 3U));
        const auto action = normalize_entity(sentence.substr(then_pos + 6U));
        if (condition.empty() || action.empty()) {
            return std::nullopt;
        }
        return PrincipleEdge{condition, PrincipleRelation::ConditionAction, action, sentence, 0.72L};
    }

    static std::optional<PrincipleEdge> parse_binary_relation(const std::string& sentence,
                                                              const std::vector<std::string_view>& markers,
                                                              PrincipleRelation relation) {
        const auto found = find_any(sentence, markers);
        if (found.first == std::string::npos) {
            return std::nullopt;
        }
        auto source = normalize_entity(sentence.substr(0, found.first));
        auto target = normalize_entity(sentence.substr(found.first + found.second.size()));
        if (source.empty() || target.empty()) {
            return std::nullopt;
        }
        return PrincipleEdge{std::move(source), relation, std::move(target), sentence, 0.68L};
    }

    void add_edge(PrincipleEdge edge) {
        if (edge.source.empty() || edge.target.empty() || edge.source == edge.target) {
            return;
        }
        const auto key = edge_key(edge.relation, edge.source, edge.target);
        const auto found = edge_index_.find(key);
        if (found != edge_index_.end() && found->second < edges_.size()) {
            edges_[found->second].strength = std::min(1.0L, edges_[found->second].strength + 0.04L);
            return;
        }
        edge_index_.emplace(key, edges_.size());
        edges_.push_back(std::move(edge));
    }

    void add_process_step(const std::string& process,
                          const std::string& step,
                          const std::string& evidence) {
        auto& steps = process_steps_[process];
        if (steps.size() >= 12U) {
            return;
        }
        if (std::find(steps.begin(), steps.end(), step) == steps.end()) {
            if (!steps.empty()) {
                add_edge({steps.back(), PrincipleRelation::Sequence, step, evidence, 0.70L});
            }
            steps.push_back(step);
        }
    }

    std::optional<std::string> best_process(const std::vector<std::string>& query_tokens) const {
        std::string best;
        long double best_score = 0.0L;
        for (const auto& [process, steps] : process_steps_) {
            if (steps.size() < 2U) {
                continue;
            }
            const auto process_tokens = tokenize_query(process);
            const long double process_score = token_overlap_score(query_tokens, process_tokens);
            long double step_score = 0.0L;
            for (const auto& step : steps) {
                step_score = std::max(step_score, token_overlap_score(query_tokens, tokenize_query(step)));
            }
            if (process_score < 0.12L && step_score < 0.25L) {
                continue;
            }
            const long double score = 0.72L * process_score + 0.28L * step_score;
            if (score > best_score) {
                best_score = score;
                best = process;
            }
        }
        if (best.empty() || best_score < 0.18L) {
            return std::nullopt;
        }
        return best;
    }

    std::optional<PrincipleAnswer> process_answer(const std::string& process) const {
        const auto found = process_steps_.find(process);
        if (found == process_steps_.end() || found->second.size() < 2U) {
            return std::nullopt;
        }
        PrincipleAnswer answer;
        std::ostringstream out;
        out << process << ": ";
        const std::size_t step_count = std::min<std::size_t>(found->second.size(), 8);
        for (std::size_t i = 0; i < step_count; ++i) {
            if (i != 0) {
                out << " -> ";
            }
            out << found->second[i];
            if (i != 0) {
                answer.edges.push_back({found->second[i - 1U], PrincipleRelation::Sequence, found->second[i], {}, 0.74L});
            }
        }
        answer.text = out.str();
        answer.confidence = std::min(0.95L, 0.48L + 0.10L * static_cast<long double>(step_count));
        return answer;
    }

    std::optional<PrincipleAnswer> causal_chain_answer(const std::vector<std::string>& query_tokens) const {
        const PrincipleEdge* best = nullptr;
        long double best_score = 0.0L;
        for (const auto& edge : edges_) {
            if (edge.relation != PrincipleRelation::Cause) {
                continue;
            }
            const long double score = token_overlap_score(query_tokens, tokenize_query(edge.target)) +
                                      0.35L * token_overlap_score(query_tokens, tokenize_query(edge.source));
            if (score > best_score) {
                best_score = score;
                best = &edge;
            }
        }
        if (best == nullptr || best_score < 0.18L) {
            return std::nullopt;
        }
        std::vector<PrincipleEdge> chain;
        chain.push_back(*best);
        while (chain.size() < 4U) {
            const auto prev = std::find_if(edges_.begin(), edges_.end(), [&](const auto& edge) {
                return edge.relation == PrincipleRelation::Cause && edge.target == chain.front().source;
            });
            if (prev == edges_.end()) {
                break;
            }
            chain.insert(chain.begin(), *prev);
        }

        PrincipleAnswer answer;
        std::ostringstream out;
        for (std::size_t i = 0; i < chain.size(); ++i) {
            if (i != 0) {
                out << " -> ";
            }
            out << chain[i].source << " -> " << chain[i].target;
        }
        answer.text = out.str();
        answer.edges = std::move(chain);
        answer.confidence = std::min(0.92L, 0.58L + 0.08L * static_cast<long double>(answer.edges.size()));
        return answer;
    }

    std::optional<PrincipleAnswer> relation_answer(const std::vector<std::string>& query_tokens,
                                                   PrincipleRelation relation,
                                                   std::string_view label) const {
        PrincipleAnswer answer;
        for (const auto& edge : edges_) {
            if (edge.relation != relation) {
                continue;
            }
            const long double score = token_overlap_score(query_tokens, tokenize_query(edge.source));
            if (score >= 0.18L) {
                answer.edges.push_back(edge);
            }
        }
        if (answer.edges.empty()) {
            return std::nullopt;
        }
        std::sort(answer.edges.begin(), answer.edges.end(), [](const auto& left, const auto& right) {
            return left.strength > right.strength;
        });
        if (answer.edges.size() > 4U) {
            answer.edges.resize(4U);
        }
        std::ostringstream out;
        out << label << ": ";
        for (std::size_t i = 0; i < answer.edges.size(); ++i) {
            if (i != 0) {
                out << "; ";
            }
            out << answer.edges[i].source << " -> " << answer.edges[i].target;
        }
        answer.text = out.str();
        answer.confidence = std::min(0.90L, 0.52L + 0.08L * static_cast<long double>(answer.edges.size()));
        return answer;
    }

    static bool asks_process(const std::vector<std::string>& tokens,
                             std::string_view lower) {
        return lower.find("process") != std::string_view::npos ||
               lower.find("how ") != std::string_view::npos ||
               has_token(tokens, "sequence") ||
               has_token(tokens, "steps") ||
               has_token(tokens, "процесс") ||
               has_token(tokens, "этапы");
    }

    static bool asks_why_or_cause(const std::vector<std::string>& tokens,
                                  std::string_view lower) {
        return lower.find("why") != std::string_view::npos ||
               lower.find("cause") != std::string_view::npos ||
               has_token(tokens, "because") ||
               has_token(tokens, "почему") ||
               has_token(tokens, "причина");
    }

    static bool asks_parts(const std::vector<std::string>& tokens,
                           std::string_view lower) {
        return lower.find("part") != std::string_view::npos ||
               lower.find("consist") != std::string_view::npos ||
               has_token(tokens, "components") ||
               has_token(tokens, "состоит") ||
               has_token(tokens, "части");
    }

    static bool asks_definition(const std::vector<std::string>& tokens,
                                 std::string_view lower) {
        return lower.find("what is") != std::string_view::npos ||
               lower.find("define") != std::string_view::npos ||
               has_token(tokens, "definition") ||
               has_token(tokens, "что") ||
               has_token(tokens, "определи");
    }

    static bool asks_contradictions(const std::vector<std::string>& tokens) {
        return has_token(tokens, "contradiction") ||
               has_token(tokens, "conflict") ||
               has_token(tokens, "inconsistent") ||
               has_token(tokens, "противоречие");
    }

    std::optional<PrincipleAnswer> transitive_causal_chain(const std::vector<std::string>& query_tokens) const {
        std::string best_target;
        long double best_score = 0.0L;
        for (const auto& edge : edges_) {
            if (edge.relation != PrincipleRelation::Cause) {
                continue;
            }
            const long double score = token_overlap_score(query_tokens, tokenize_query(edge.target));
            if (score > best_score) {
                best_score = score;
                best_target = edge.target;
            }
        }
        if (best_target.empty() || best_score < 0.18L) {
            return std::nullopt;
        }

        std::vector<std::string> chain;
        std::string current = best_target;
        std::unordered_set<std::string> visited;
        for (std::size_t depth = 0; depth < 6U && visited.insert(current).second; ++depth) {
            chain.push_back(current);
            const auto prev = std::find_if(edges_.begin(), edges_.end(), [&](const auto& e) {
                return e.relation == PrincipleRelation::Cause && e.target == current;
            });
            if (prev == edges_.end()) {
                break;
            }
            current = prev->source;
        }
        std::reverse(chain.begin(), chain.end());
        if (chain.size() < 2U) {
            return std::nullopt;
        }

        PrincipleAnswer answer;
        std::ostringstream out;
        for (std::size_t i = 0; i < chain.size(); ++i) {
            if (i != 0) {
                out << " -> ";
            }
            out << chain[i];
            if (i != 0) {
                const auto cause_edge = std::find_if(edges_.begin(), edges_.end(), [&](const auto& e) {
                    return e.relation == PrincipleRelation::Cause &&
                           e.source == chain[i - 1] && e.target == chain[i];
                });
                if (cause_edge != edges_.end()) {
                    answer.edges.push_back(*cause_edge);
                }
            }
        }
        answer.text = out.str();
        answer.confidence = std::min(0.92L, 0.48L + 0.08L * static_cast<long double>(chain.size()));
        return answer;
    }

    std::optional<PrincipleAnswer> detect_contradictions(const std::vector<std::string>& query_tokens) const {
        std::string best_entity;
        long double best_score = 0.0L;
        for (const auto& edge : edges_) {
            if (edge.relation != PrincipleRelation::Definition) {
                continue;
            }
            const long double score = token_overlap_score(query_tokens, tokenize_query(edge.source));
            if (score > best_score) {
                best_score = score;
                best_entity = edge.source;
            }
        }
        if (best_entity.empty() || best_score < 0.18L) {
            return std::nullopt;
        }

        std::vector<PrincipleEdge> conflicting;
        for (const auto& edge : edges_) {
            if (edge.relation == PrincipleRelation::Definition && edge.source == best_entity) {
                conflicting.push_back(edge);
            }
        }
        if (conflicting.size() < 2U) {
            return std::nullopt;
        }
        std::sort(conflicting.begin(), conflicting.end(), [](const auto& a, const auto& b) {
            return a.strength > b.strength;
        });
        if (conflicting.size() > 4U) {
            conflicting.resize(4U);
        }

        PrincipleAnswer answer;
        std::ostringstream out;
        out << "contradiction in definitions of '" << best_entity << "': ";
        for (std::size_t i = 0; i < conflicting.size(); ++i) {
            if (i != 0) {
                out << " vs ";
            }
            out << conflicting[i].source << " is " << conflicting[i].target;
        }
        answer.text = out.str();
        answer.edges = std::move(conflicting);
        answer.confidence = 0.78L;
        return answer;
    }

    static bool has_token(const std::vector<std::string>& tokens,
                          std::string_view token) {
        return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
    }

    static long double token_overlap_score(const std::vector<std::string>& query,
                                           const std::vector<std::string>& candidate) {
        if (query.empty() || candidate.empty()) {
            return 0.0L;
        }
        std::size_t shared = 0;
        for (const auto& token : query) {
            if (token.size() < 3U || is_stopword(token)) {
                continue;
            }
            if (std::find(candidate.begin(), candidate.end(), token) != candidate.end()) {
                ++shared;
            }
        }
        return static_cast<long double>(shared) /
               static_cast<long double>(std::max<std::size_t>(1, candidate.size()));
    }

    static bool is_stopword(std::string_view token) {
        return token == "the" || token == "and" || token == "for" || token == "with" ||
               token == "what" || token == "why" || token == "how" || token == "does" ||
               token == "are" || token == "is" || token == "process" || token == "explain" ||
               token == "answer" || token == "describe" || token == "это" || token == "как" ||
               token == "что" || token == "почему";
    }

    static std::pair<std::size_t, std::string_view> find_any(const std::string& text,
                                                             const std::vector<std::string_view>& markers) {
        std::pair<std::size_t, std::string_view> best{std::string::npos, {}};
        for (auto marker : markers) {
            const auto pos = text.find(marker);
            if (pos != std::string::npos && pos < best.first) {
                best = {pos, marker};
            }
        }
        return best;
    }

    static std::string_view leading_sequence_marker(std::string_view text) {
        for (auto marker : {std::string_view("then "), std::string_view("next "),
                            std::string_view("after "), std::string_view("затем "),
                            std::string_view("потом "), std::string_view("после ")}) {
            if (text.rfind(marker, 0) == 0) {
                return marker;
            }
        }
        return {};
    }

    static std::string normalize_phrase(std::string_view text) {
        std::string out;
        out.reserve(text.size());
        bool last_space = true;
        for (const unsigned char ch : text) {
            if (std::isalnum(ch) != 0 || static_cast<unsigned char>(ch) >= 128U) {
                out.push_back(static_cast<char>(std::tolower(ch)));
                last_space = false;
            } else if (!last_space) {
                out.push_back(' ');
                last_space = true;
            }
        }
        return trim(std::move(out));
    }

    static std::string normalize_entity(std::string text) {
        text = trim(std::move(text));
        const std::vector<std::string_view> prefixes{
            "the ", "a ", "an ", "this ", "that ", "then ", "next ", "finally ",
            "затем ", "потом ", "это "};
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto prefix : prefixes) {
                if (text.rfind(prefix, 0) == 0) {
                    text.erase(0, prefix.size());
                    text = trim(std::move(text));
                    changed = true;
                }
            }
        }
        return text;
    }

    static std::string remove_trailing_verbs(std::string text) {
        const std::vector<std::string_view> suffixes{
            " occurs", " occur", " happens", " happen", " begins", " begin"};
        for (auto suffix : suffixes) {
            if (text.size() > suffix.size() &&
                text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0) {
                text.resize(text.size() - suffix.size());
                return trim(std::move(text));
            }
        }
        return text;
    }

    static std::string edge_key(PrincipleRelation relation,
                                const std::string& source,
                                const std::string& target) {
        return std::to_string(static_cast<std::uint32_t>(relation)) + "\t" + source + "\t" + target;
    }

    static std::string trim(std::string text) {
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return {};
        }
        const auto last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1U);
    }

    std::vector<PrincipleEdge> edges_;
    std::unordered_map<std::string, std::size_t> edge_index_;
    std::unordered_map<std::string, std::vector<std::string>> process_steps_;
};

} // namespace dzeta
