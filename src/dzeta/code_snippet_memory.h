#pragma once

#include "mentalese_core.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dzeta {

struct CodeSnippet {
    std::string code;
    std::string function_name;
    std::vector<std::string> name_tokens;
    std::vector<std::string> tokens;
    std::vector<std::string> structure_tokens;
    std::vector<long double> structure_vector;
    std::vector<std::string> parameters;
    std::string return_expression;
    bool indented_definition = false;
    bool private_or_internal = false;
    bool method_like = false;
    std::size_t observations = 1;
    long double reliability = 0.6L;
};

struct CodeSnippetHit {
    std::string code;
    long double score = 0.0L;
};

struct CodeOperationLink {
    std::string operation;
    std::string expression_template;
    long double strength = 0.0L;
};

struct CodePredicateMotif {
    std::string trigger;
    std::string predicate_template;
    long double strength = 0.0L;
};

struct CodeTransformMotif {
    std::string trigger;
    std::string expression_template;
    long double strength = 0.0L;
};

struct CodeReduceMotif {
    std::string trigger;
    std::string initial_value;
    std::string statement_template;
    long double strength = 0.0L;
};

class CodeSnippetMemory {
public:
    explicit CodeSnippetMemory(std::size_t max_snippets = 2048)
        : max_snippets_(std::max<std::size_t>(32, max_snippets)) {}

    void observe_text(std::string_view text) {
        for (auto snippet : prepare_snippets(text, 24)) {
            observe_prepared(std::move(snippet));
        }
        compact();
    }

    void observe_prepared(CodeSnippet snippet) {
        observe_prepared_impl(std::move(snippet), false, false);
    }

    void observe_prepared_impl(CodeSnippet snippet, bool defer_token_index, bool defer_operation_links) {
        if (snippet.code.empty()) {
            return;
        }
        if (static_literal_dump(snippet) || non_algorithmic_bulk_snippet(snippet)) {
            return;
        }
        if (defer_operation_links) {
            operation_links_dirty_ = true;
        } else {
            learn_operation_links(snippet);
        }
        const auto index = snippets_.size();
        snippets_.push_back(std::move(snippet));
        if (defer_token_index) {
            token_index_dirty_ = true;
        } else {
            index_snippet(index, snippets_.back());
        }
        reinforce_structure_organs(snippets_.back());
        if (snippets_.size() > max_snippets_ * 2U) {
            compact();
        }
    }

    void restore_serialized_snippet(std::string code,
                                    std::size_t observations,
                                    long double reliability) {
        auto parsed = prepare_snippets(code, 1);
        if (parsed.empty()) {
            return;
        }
        auto snippet = std::move(parsed.front());
        snippet.observations = std::max<std::size_t>(1, observations);
        snippet.reliability = std::clamp(reliability, 0.0L, 1.0L);
        observe_prepared(std::move(snippet));
    }

    void restore_prepared_snippet(CodeSnippet snippet,
                                  bool defer_token_index = false,
                                  bool defer_operation_links = false,
                                  bool recompute_structure = true) {
        if (snippet.code.empty()) {
            return;
        }
        if (snippet.function_name.empty()) {
            snippet.function_name = extract_function_name(snippet.code);
        }
        if (snippet.name_tokens.empty()) {
            snippet.name_tokens = normalize_tokens(tokenize_query(snippet.function_name));
        }
        if (snippet.tokens.empty()) {
            snippet.tokens = normalize_tokens(tokenize_query(strip_python_non_code_text(snippet.code)));
        }
        if (snippet.structure_tokens.empty()) {
            snippet.structure_tokens = extract_structure_tokens(snippet.code);
        } else if (recompute_structure) {
            for (auto token : extract_structure_tokens(snippet.code)) {
                if (std::find(snippet.structure_tokens.begin(), snippet.structure_tokens.end(), token) ==
                    snippet.structure_tokens.end()) {
                    snippet.structure_tokens.push_back(std::move(token));
                }
            }
        }
        if (snippet.structure_vector.empty()) {
            snippet.structure_vector = extract_structure_vector(snippet.code);
        }
        if (snippet.parameters.empty()) {
            snippet.parameters = extract_parameters(snippet.code);
        }
        if (snippet.return_expression.empty()) {
            snippet.return_expression = extract_return_expression(snippet.code);
        }
        snippet.observations = std::max<std::size_t>(1, snippet.observations);
        snippet.reliability = std::clamp(snippet.reliability, 0.0L, 1.0L);
        observe_prepared_impl(std::move(snippet), defer_token_index, defer_operation_links);
    }

    void restore_structure_organs(std::vector<std::string> structure_tokens,
                                  std::size_t observations,
                                  long double reliability) {
        reinforce_structure_organs_from_tokens(structure_tokens,
                                               std::max<std::size_t>(1, observations),
                                               std::clamp(reliability, 0.0L, 1.0L));
    }

    void restore_organ_strength(std::string organ, long double strength) {
        if (!organ.empty() && std::isfinite(static_cast<double>(strength))) {
            organ_strengths_[std::move(organ)] += std::max(0.0L, strength);
        }
    }

    std::size_t size() const noexcept {
        return snippets_.size();
    }

    const std::vector<CodeSnippet>& snippets() const noexcept {
        return snippets_;
    }

    const std::unordered_map<std::string, long double>& organ_strengths() const noexcept {
        return organ_strengths_;
    }

    const std::vector<CodePredicateMotif>& predicate_motifs() const noexcept {
        return predicate_motifs_;
    }

    const std::vector<CodeTransformMotif>& transform_motifs() const noexcept {
        return transform_motifs_;
    }

    const std::vector<CodeReduceMotif>& reduce_motifs() const noexcept {
        return reduce_motifs_;
    }

    long double count_if_trajectory_strength() const noexcept {
        return count_if_trajectory_strength_;
    }

    long double collect_if_trajectory_strength() const noexcept {
        return collect_if_trajectory_strength_;
    }

    long double map_transform_trajectory_strength() const noexcept {
        return map_transform_trajectory_strength_;
    }

    long double reduce_accumulator_trajectory_strength() const noexcept {
        return reduce_accumulator_trajectory_strength_;
    }

    void restore_predicate_motif(std::string trigger,
                                 std::string predicate_template,
                                 long double strength) {
        if (trigger.empty() || predicate_template.empty() ||
            !std::isfinite(static_cast<double>(strength))) {
            return;
        }
        add_predicate_motif({std::move(trigger),
                             std::move(predicate_template),
                             std::max(0.0L, strength)});
    }

    void restore_transform_motif(std::string trigger,
                                 std::string expression_template,
                                 long double strength) {
        if (trigger.empty() || expression_template.empty() ||
            !std::isfinite(static_cast<double>(strength))) {
            return;
        }
        add_transform_motif({std::move(trigger),
                             std::move(expression_template),
                             std::max(0.0L, strength)});
    }

    void restore_reduce_motif(std::string trigger,
                              std::string initial_value,
                              std::string statement_template,
                              long double strength) {
        if (trigger.empty() || initial_value.empty() || statement_template.empty() ||
            !std::isfinite(static_cast<double>(strength))) {
            return;
        }
        add_reduce_motif({std::move(trigger),
                          std::move(initial_value),
                          std::move(statement_template),
                          std::max(0.0L, strength)});
    }

    void restore_count_if_trajectory_strength(long double strength) {
        if (std::isfinite(static_cast<double>(strength))) {
            count_if_trajectory_strength_ =
                std::max(count_if_trajectory_strength_, std::max(0.0L, strength));
        }
    }

    void restore_collect_if_trajectory_strength(long double strength) {
        if (std::isfinite(static_cast<double>(strength))) {
            collect_if_trajectory_strength_ =
                std::max(collect_if_trajectory_strength_, std::max(0.0L, strength));
        }
    }

    void restore_map_transform_trajectory_strength(long double strength) {
        if (std::isfinite(static_cast<double>(strength))) {
            map_transform_trajectory_strength_ =
                std::max(map_transform_trajectory_strength_, std::max(0.0L, strength));
        }
    }

    void restore_reduce_accumulator_trajectory_strength(long double strength) {
        if (std::isfinite(static_cast<double>(strength))) {
            reduce_accumulator_trajectory_strength_ =
                std::max(reduce_accumulator_trajectory_strength_, std::max(0.0L, strength));
        }
    }

    void merge_code_organs_from(const CodeSnippetMemory& other) {
        for (const auto& [organ, strength] : other.organ_strengths()) {
            restore_organ_strength(organ, strength);
        }
        restore_count_if_trajectory_strength(other.count_if_trajectory_strength());
        restore_collect_if_trajectory_strength(other.collect_if_trajectory_strength());
        restore_map_transform_trajectory_strength(other.map_transform_trajectory_strength());
        restore_reduce_accumulator_trajectory_strength(other.reduce_accumulator_trajectory_strength());
        for (const auto& motif : other.predicate_motifs()) {
            restore_predicate_motif(motif.trigger, motif.predicate_template, motif.strength);
        }
        for (const auto& motif : other.transform_motifs()) {
            restore_transform_motif(motif.trigger, motif.expression_template, motif.strength);
        }
        for (const auto& motif : other.reduce_motifs()) {
            restore_reduce_motif(motif.trigger, motif.initial_value, motif.statement_template, motif.strength);
        }
    }

    void reserve(std::size_t count) {
        snippets_.reserve(count);
        token_index_.reserve(std::max<std::size_t>(64, count / 2U));
        organ_strengths_.reserve(32);
        operation_links_.reserve(64);
        predicate_motifs_.reserve(64);
        transform_motifs_.reserve(64);
        reduce_motifs_.reserve(64);
    }

    static bool prompt_is_complex_algorithm(std::string_view prompt) {
        return structural_algorithm_query(salient_query_tokens(tokenize_query(prompt)));
    }

    static bool prompt_can_use_latent_algorithm_organs(std::string_view prompt) {
        const auto salient = salient_query_tokens(tokenize_query(prompt));
        const auto has = [&](std::string_view token) {
            return std::find(salient.begin(), salient.end(), token) != salient.end();
        };
        return has("dijkstra") || has("shortest") || has("path") ||
               has("priority") || has("tarjan") || has("component") ||
               has("connected") || has("cache") || has("lru") ||
               has("eviction") || (has("binary") && has("search")) ||
               (has("merge") && has("sort")) || (has("two") && has("sum")) ||
               has("pair") || has("indice") || has("index") ||
               has("bfs") || has("breadth") || has("unweighted") ||
               has("unweight") || (has("sliding") && has("window")) ||
               (has("slid") && has("window")) || has("disjoint") ||
               (has("union") && has("find")) || has("topological") ||
               (has("dependency") && has("graph"));
    }

    static long double prompt_code_resonance(std::string_view prompt, std::string_view code) {
        const auto query = query_structure_vector(salient_query_tokens(tokenize_query(prompt)));
        const auto candidate = extract_structure_vector(std::string(code));
        return vector_similarity(candidate, query);
    }

    std::vector<CodeSnippetHit> retrieve(std::string_view prompt, std::size_t limit) const {
        const auto query_tokens = tokenize_query(prompt);
        const auto salient = salient_query_tokens(query_tokens);
        const auto candidates = indexed_candidates(salient);
        std::vector<CodeSnippetHit> hits;
        const auto score_one = [&](const CodeSnippet& snippet) {
            const long double score = score_snippet(snippet, query_tokens);
            if (score >= 0.22L) {
                hits.push_back({snippet.code, std::min(1.0L, score)});
            }
        };
        if (candidates.empty()) {
            for (const auto& snippet : snippets_) {
                score_one(snippet);
            }
        } else {
            for (const auto index : candidates) {
                if (index < snippets_.size()) {
                    score_one(snippets_[index]);
                }
            }
        }
        std::sort(hits.begin(), hits.end(), [](const auto& left, const auto& right) {
            return left.score > right.score;
        });
        if (hits.size() > limit) {
            hits.resize(limit);
        }
        return hits;
    }

    std::optional<std::string> synthesize(std::string_view prompt) const {
        const auto query_tokens = tokenize_query(prompt);
        const auto target_tokens = target_function_tokens(query_tokens);
        if (target_tokens.empty()) {
            return std::nullopt;
        }
        const auto salient = salient_query_tokens(query_tokens);
        if (structural_algorithm_query(salient)) {
            return std::nullopt;
        }
        const CodeSnippet* best = nullptr;
        long double best_score = 0.0L;
        const auto candidates = indexed_candidates(salient);
        const auto consider = [&](const CodeSnippet& snippet) {
            if (snippet.return_expression.empty() ||
                snippet.private_or_internal ||
                snippet.method_like ||
                snippet.indented_definition) {
                return;
            }
            const long double score = score_snippet(snippet, query_tokens);
            if (score > best_score) {
                best_score = score;
                best = &snippet;
            }
        };
        if (candidates.empty()) {
            for (const auto& snippet : snippets_) {
                consider(snippet);
            }
        } else {
            for (const auto index : candidates) {
                if (index < snippets_.size()) {
                    consider(snippets_[index]);
                }
            }
        }
        if (best == nullptr || best_score < 0.30L) {
            if (const auto trajectory = synthesize_from_trajectory_links(query_tokens, target_tokens)) {
                return trajectory;
            }
            return synthesize_from_operation_links(query_tokens, target_tokens);
        }
        if (under_covers_structural_constraints(best->tokens, salient)) {
            return std::nullopt;
        }
        if (structural_algorithm_query(salient) && !structural_candidate_body(*best)) {
            return std::nullopt;
        }
        const auto function_name = join_identifier(target_tokens);
        if (best->function_name == function_name) {
            return std::nullopt;
        }
        const auto target_parameter = infer_target_parameter(query_tokens);
        const auto source_parameter = best->parameters.empty() ? std::string("value") : best->parameters.front();
        if (!identifier_present(best->return_expression, source_parameter)) {
            if (const auto trajectory = synthesize_from_trajectory_links(query_tokens, target_tokens)) {
                return trajectory;
            }
            return synthesize_from_operation_links(query_tokens, target_tokens);
        }
        if (!return_expression_is_self_contained(best->return_expression, source_parameter)) {
            if (const auto trajectory = synthesize_from_trajectory_links(query_tokens, target_tokens)) {
                return trajectory;
            }
            return synthesize_from_operation_links(query_tokens, target_tokens);
        }
        auto expression = replace_identifier(best->return_expression, source_parameter, target_parameter);
        if (expression.empty()) {
            if (const auto trajectory = synthesize_from_trajectory_links(query_tokens, target_tokens)) {
                return trajectory;
            }
            return synthesize_from_operation_links(query_tokens, target_tokens);
        }
        return "def " + function_name + "(" + target_parameter + "):\n    return " + expression + "\n";
    }

    std::optional<std::string> synthesize_latent_algorithm(std::string_view prompt) const {
        const auto query_tokens = tokenize_query(prompt);
        const auto salient = salient_query_tokens(query_tokens);
        if (!structural_algorithm_query(salient)) {
            return std::nullopt;
        }
        return synthesize_best_latent_algorithm(query_tokens, salient);
    }

    void compact() {
        if (snippets_.size() <= max_snippets_) {
            return;
        }
        std::sort(snippets_.begin(), snippets_.end(), [](const auto& left, const auto& right) {
            if (left.reliability == right.reliability) {
                return left.observations > right.observations;
            }
            return left.reliability > right.reliability;
        });
        snippets_.resize(max_snippets_);
        operation_links_.clear();
        predicate_motifs_.clear();
        transform_motifs_.clear();
        reduce_motifs_.clear();
        count_if_trajectory_strength_ = 0.0L;
        collect_if_trajectory_strength_ = 0.0L;
        map_transform_trajectory_strength_ = 0.0L;
        reduce_accumulator_trajectory_strength_ = 0.0L;
        operation_links_dirty_ = true;
        rebuild_token_index();
    }

    static std::vector<CodeSnippet> prepare_snippets(std::string_view text, std::size_t limit) {
        std::vector<CodeSnippet> out;
        std::stringstream stream{std::string(text)};
        std::string line;
        std::string current;
        bool capturing = false;
        std::size_t definition_indent = 0;
        while (std::getline(stream, line)) {
            const auto trimmed = ltrim(line);
            const bool starts_def = trimmed.rfind("def ", 0) == 0;
            const bool starts_class = trimmed.rfind("class ", 0) == 0;
            const std::size_t indent = indentation_width(line);
            const bool starts_peer_definition = (starts_def || starts_class) &&
                                                (!capturing || indent <= definition_indent);
            const bool ends_current_definition = capturing &&
                                                 !trimmed.empty() &&
                                                 !starts_def &&
                                                 !starts_class &&
                                                 indent <= definition_indent &&
                                                 current.find('\n') != std::string::npos;
            if (ends_current_definition) {
                flush_current(current, out, limit);
                capturing = false;
                if (out.size() >= limit) {
                    return out;
                }
            }
            if (starts_peer_definition) {
                flush_current(current, out, limit);
                if (out.size() >= limit) {
                    return out;
                }
                capturing = starts_def;
                definition_indent = indent;
                current.clear();
            }
            if (capturing) {
                current += line;
                current.push_back('\n');
                if (current.size() > 24000U) {
                    flush_current(current, out, limit);
                    capturing = false;
                    if (out.size() >= limit) {
                        return out;
                    }
                }
            }
        }
        flush_current(current, out, limit);
        return out;
    }

private:
    void index_snippet(std::size_t index, const CodeSnippet& snippet) const {
        std::vector<std::string> keys;
        keys.reserve(snippet.name_tokens.size());
        for (const auto& token : snippet.name_tokens) {
            if (indexable_token(token) && std::find(keys.begin(), keys.end(), token) == keys.end()) {
                keys.push_back(token);
            }
        }
        for (const auto& token : snippet.tokens) {
            if (structural_signature_token(token) &&
                std::find(keys.begin(), keys.end(), token) == keys.end()) {
                keys.push_back(token);
            }
        }
        for (const auto& token : snippet.structure_tokens) {
            if (std::find(keys.begin(), keys.end(), token) == keys.end()) {
                keys.push_back(token);
            }
        }
        for (const auto& token : keys) {
            token_index_[token].push_back(index);
        }
    }

    void rebuild_token_index() {
        token_index_.clear();
        organ_strengths_.clear();
        for (std::size_t index = 0; index < snippets_.size(); ++index) {
            index_snippet(index, snippets_[index]);
            reinforce_structure_organs(snippets_[index]);
        }
        token_index_dirty_ = false;
    }

    void ensure_token_index() const {
        if (!token_index_dirty_) {
            return;
        }
        token_index_.clear();
        for (std::size_t index = 0; index < snippets_.size(); ++index) {
            index_snippet(index, snippets_[index]);
        }
        token_index_dirty_ = false;
    }

    void reinforce_structure_organs(const CodeSnippet& snippet) {
        std::vector<std::string> unique = snippet.structure_tokens;
        if (structural_candidate_body(snippet) &&
            std::find(unique.begin(), unique.end(), "algorithmic_body") == unique.end()) {
            unique.push_back("algorithmic_body");
        }
        reinforce_structure_organs_from_tokens(unique, snippet.observations, snippet.reliability);
    }

    void reinforce_structure_organs_from_tokens(const std::vector<std::string>& unique,
                                                std::size_t observations,
                                                long double reliability) {
        const long double weight =
            std::clamp(reliability, 0.05L, 1.0L) *
            std::log1pl(static_cast<long double>(std::max<std::size_t>(1, observations)));
        for (const auto& token : unique) {
            organ_strengths_[token] += std::max(0.05L, weight);
        }
    }

    long double organ_strength(std::string_view token) const {
        const auto found = organ_strengths_.find(std::string(token));
        return found == organ_strengths_.end() ? 0.0L : found->second;
    }

    bool learned_organs_support(std::initializer_list<std::string_view> required) const {
        if (required.size() == 0U || (snippets_.empty() && organ_strengths_.empty())) {
            return false;
        }
        long double total = 0.0L;
        for (const auto token : required) {
            const auto strength = organ_strength(token);
            if (strength < 0.30L) {
                return false;
            }
            total += std::min(4.0L, strength);
        }
        return total >= 0.48L * static_cast<long double>(required.size());
    }

    std::vector<std::size_t> indexed_candidates(const std::vector<std::string>& salient) const {
        std::vector<std::size_t> out;
        ensure_token_index();
        if (salient.empty() || snippets_.empty() || token_index_.empty()) {
            return out;
        }
        std::vector<unsigned char> seen(snippets_.size(), 0);
        for (const auto& token : salient) {
            const auto found = token_index_.find(token);
            if (found == token_index_.end()) {
                continue;
            }
            for (const auto index : found->second) {
                if (index < seen.size() && seen[index] == 0) {
                    seen[index] = 1;
                    out.push_back(index);
                }
            }
        }
        return out;
    }

    static bool indexable_token(std::string_view token) {
        return token.size() >= 3U && !is_code_query_stopword(token);
    }

    static bool structural_signature_token(std::string_view token) {
        return token == "topological" || token == "dependency" || token == "graph" ||
               token == "cycle" || token == "edge" || token == "node" ||
               token == "indegree" || token == "in_degree" || token == "neighbor" ||
               token == "successor" || token == "predecessor" || token == "upstream" ||
               token == "downstream" || token == "visited" || token == "visit" ||
               token == "tarjan" || token == "dijkstra" || token == "shortest" ||
               token == "path" || token == "priority" || token == "queue" ||
               token == "component" || token == "connected" || token == "strongly" ||
               token == "weighted" || token == "cache" ||
               token == "eviction" || token == "ordered" || token == "dictionary" ||
               token == "binary" || token == "search" || token == "merge" ||
               token == "sort" || token == "two" || token == "sum" ||
               token == "pair" || token == "index" || token == "indice" ||
               token == "target" || token == "complement" || token == "bfs" ||
               token == "breadth" || token == "unweighted" || token == "unweight" ||
               token == "sliding" || token == "slid" ||
               token == "window" || token == "maximum" || token == "disjoint" ||
               token == "union" || token == "find" || token == "rank" ||
               token == "compression";
    }

    static std::vector<std::string> extract_structure_tokens(const std::string& code) {
        const auto lower = lower_copy(code);
        std::vector<std::string> out;
        const auto add = [&](std::string token) {
            if (std::find(out.begin(), out.end(), token) == out.end()) {
                out.push_back(std::move(token));
            }
        };
        if (lower.find("\n    for ") != std::string::npos || lower.find(" for ") != std::string::npos) {
            add("for_loop");
        }
        if (lower.find("\n    while ") != std::string::npos || lower.find(" while ") != std::string::npos) {
            add("while_loop");
        }
        if (lower.find("+=") != std::string::npos || lower.find("-=") != std::string::npos) {
            add("accumulator");
        }
        if (lower.find(".append(") != std::string::npos || lower.find(".extend(") != std::string::npos) {
            add("sequence_builder");
        }
        if (lower.find(".pop(") != std::string::npos || lower.find("popleft(") != std::string::npos) {
            add("frontier_pop");
        }
        if (lower.find(".pop(0)") != std::string::npos || lower.find("popleft(") != std::string::npos) {
            add("fifo_queue");
        }
        if (lower.find("raise ") != std::string::npos) {
            add("exception_guard");
        }
        if (lower.find("yield ") != std::string::npos) {
            add("generator_flow");
        }
        if (lower.find("heapq") != std::string::npos || lower.find("heappush") != std::string::npos ||
            lower.find("heappop") != std::string::npos ||
            lower.find("priority") != std::string::npos ||
            lower.find("frontier.sort") != std::string::npos) {
            add("priority_queue");
        }
        if (lower.find("set(") != std::string::npos || lower.find("difference_update") != std::string::npos ||
            lower.find(".add(") != std::string::npos) {
            add("set_state");
        }
        if (lower.find("{") != std::string::npos || lower.find(".items(") != std::string::npos ||
            lower.find(".get(") != std::string::npos || lower.find("defaultdict") != std::string::npos) {
            add("mapping_state");
        }
        if (lower.find("graph") != std::string::npos || lower.find("edge") != std::string::npos ||
            lower.find("node") != std::string::npos || lower.find("neighbor") != std::string::npos ||
            lower.find("successor") != std::string::npos || lower.find("predecessor") != std::string::npos) {
            add("graph_traversal");
        }
        if (lower.find("previous[") != std::string::npos ||
            lower.find("node = previous") != std::string::npos) {
            add("path_reconstruction");
        }
        if (lower.find("indegree") != std::string::npos || lower.find("in_degree") != std::string::npos) {
            add("dependency_ordering");
        }
        if (lower.find("visited") != std::string::npos || lower.find("visit(") != std::string::npos ||
            lower.find("dfs") != std::string::npos) {
            add("visited_search");
        }
        if (lower.find("cache") != std::string::npos || lower.find("ordered") != std::string::npos ||
            lower.find("move_to_end") != std::string::npos || lower.find("popitem") != std::string::npos) {
            add("cache_eviction");
        }
        if ((lower.find("low") != std::string::npos && lower.find("high") != std::string::npos &&
             lower.find("mid") != std::string::npos) ||
            lower.find("// 2") != std::string::npos) {
            add("bounds_search");
        }
        if (lower.find("enumerate(") != std::string::npos &&
            (lower.find("seen[") != std::string::npos || lower.find(" in seen") != std::string::npos)) {
            add("pair_index_map");
        }
        if (lower.find("complement") != std::string::npos ||
            (lower.find("target - ") != std::string::npos && lower.find(" in seen") != std::string::npos)) {
            add("complement_search");
        }
        if (lower.find("window") != std::string::npos &&
            (lower.find("[start:start +") != std::string::npos ||
             lower.find("[start : start +") != std::string::npos)) {
            add("sliding_window");
        }
        if ((lower.find("deque(") != std::string::npos || lower.find("collections.deque") != std::string::npos) &&
            lower.find("popleft(") != std::string::npos &&
            (lower.find("queue[-1]") != std::string::npos || lower.find("window[-1]") != std::string::npos)) {
            add("monotonic_window");
            add("sliding_window");
        }
        if (lower.find("range(0, len(") != std::string::npos &&
            (lower.find(" - width + 1") != std::string::npos ||
             lower.find(" - k + 1") != std::string::npos)) {
            add("range_window");
        }
        if (lower.find("max(window)") != std::string::npos ||
            lower.find("min(window)") != std::string::npos ||
            lower.find("sum(window)") != std::string::npos ||
            lower.find("values[queue[0]]") != std::string::npos) {
            add("window_aggregate");
        }
        if (lower.find("parent[") != std::string::npos &&
            lower.find("rank") != std::string::npos &&
            lower.find("def find") != std::string::npos) {
            add("disjoint_set");
            add("path_compression");
        }
        if (lower.find("[:mid]") != std::string::npos || lower.find("[mid:]") != std::string::npos ||
            lower.find("split") != std::string::npos || lower.find("recombine") != std::string::npos) {
            add("divide_conquer");
        }
        if ((lower.find("while i < len") != std::string::npos &&
             lower.find("j < len") != std::string::npos) ||
            (lower.find("left[i]") != std::string::npos && lower.find("right[j]") != std::string::npos)) {
            add("two_pointer_merge");
        }
        if (lower.find("return ") != std::string::npos) {
            add("returns_value");
        }
        return out;
    }

    static std::vector<long double> extract_structure_vector(const std::string& code) {
        const auto lower = lower_copy(code);
        std::vector<long double> vector(24, 0.0L);
        const auto add_if = [&](std::size_t index, std::string_view marker, long double value) {
            if (lower.find(marker) != std::string::npos) {
                vector[index] += value;
            }
        };
        add_if(0, " for ", 1.0L);
        add_if(0, "\n    for ", 1.0L);
        add_if(1, " while ", 1.0L);
        add_if(1, "\n    while ", 1.0L);
        add_if(2, "+=", 1.0L);
        add_if(2, "-=", 0.8L);
        add_if(3, ".append(", 1.0L);
        add_if(3, ".extend(", 0.7L);
        add_if(4, ".pop(", 1.0L);
        add_if(4, "popleft(", 1.0L);
        add_if(5, "{", 0.6L);
        add_if(5, ".items(", 1.0L);
        add_if(5, ".get(", 0.8L);
        add_if(5, "defaultdict", 1.0L);
        add_if(6, "set(", 0.8L);
        add_if(6, ".add(", 0.8L);
        add_if(6, "difference_update", 1.0L);
        add_if(7, "raise ", 1.0L);
        add_if(8, "heapq", 1.0L);
        add_if(8, "heappush", 1.0L);
        add_if(8, "heappop", 1.0L);
        add_if(8, "priority", 0.9L);
        add_if(8, "frontier.sort", 0.8L);
        add_if(9, "graph", 0.8L);
        add_if(9, "node", 0.7L);
        add_if(9, "edge", 0.7L);
        add_if(9, "neighbor", 0.8L);
        add_if(10, "indegree", 1.0L);
        add_if(10, "in_degree", 1.0L);
        add_if(11, "visited", 1.0L);
        add_if(11, "visit(", 0.8L);
        add_if(11, "dfs", 0.8L);
        add_if(12, "cache", 1.0L);
        add_if(12, "move_to_end", 1.0L);
        add_if(12, "popitem", 1.0L);
        add_if(13, "yield ", 1.0L);
        add_if(14, "return ", 0.5L);
        add_if(15, "def ", 0.3L);
        add_if(16, "enumerate(", 1.0L);
        add_if(16, "seen[", 0.9L);
        add_if(17, "complement", 1.0L);
        add_if(17, "target - ", 0.8L);
        add_if(18, ".pop(0)", 1.0L);
        add_if(18, "popleft(", 1.0L);
        add_if(19, "previous[", 1.0L);
        add_if(19, "path.reverse", 0.8L);
        add_if(20, "window", 1.0L);
        add_if(20, "[start:start +", 1.0L);
        add_if(21, "range(0, len", 1.0L);
        add_if(22, "max(window)", 1.0L);
        add_if(22, "sum(window)", 0.8L);
        add_if(23, "unweighted", 0.6L);
        normalize_vector(vector);
        return vector;
    }

    static void normalize_vector(std::vector<long double>& vector) {
        long double norm = 0.0L;
        for (const auto value : vector) {
            norm += value * value;
        }
        norm = std::sqrt(norm);
        if (norm <= 1.0e-12L) {
            return;
        }
        for (auto& value : vector) {
            value /= norm;
        }
    }

    void learn_operation_links(const CodeSnippet& snippet) const {
        const auto lower = lower_copy(snippet.code);
        if (lower.find("sorted(") != std::string::npos &&
            (has_token(snippet.tokens, "sort") || lower.find("sort") != std::string::npos)) {
            add_operation_link({"sort", "sorted({arg})", 0.76L});
        }
        if (lower.find("[::-1]") != std::string::npos &&
            (has_token(snippet.tokens, "reverse") || lower.find("reverse") != std::string::npos)) {
            add_operation_link({"reverse", "{arg}[::-1]", 0.76L});
        }
        learn_generic_return_motif(snippet);
        learn_predicate_motifs(snippet);
        learn_count_if_trajectory(snippet);
        learn_collect_if_trajectory(snippet);
        learn_map_transform_trajectory(snippet);
        learn_reduce_accumulator_trajectory(snippet);
    }

    void add_operation_link(CodeOperationLink link) const {
        const auto found = std::find_if(operation_links_.begin(), operation_links_.end(), [&](const auto& item) {
            return item.operation == link.operation && item.expression_template == link.expression_template;
        });
        if (found != operation_links_.end()) {
            found->strength = std::min(1.0L, found->strength + 0.04L);
            return;
        }
        operation_links_.push_back(std::move(link));
    }

    void learn_generic_return_motif(const CodeSnippet& snippet) const {
        if (snippet.return_expression.empty() || snippet.parameters.empty()) {
            return;
        }
        const auto source_parameter = return_parameter(snippet.parameters, snippet.return_expression);
        if (source_parameter.empty()) {
            return;
        }
        auto expression_template = replace_identifier(snippet.return_expression, source_parameter, "{arg}");
        for (const auto& operation : infer_operations_from_expression(snippet.return_expression)) {
            const auto canonical_template = canonical_operation_template(operation, expression_template);
            if (canonical_template.empty()) {
                continue;
            }
            add_operation_link({operation, canonical_template, 0.70L});
            const auto normalized_operation = normalize_token(std::string(operation));
            if (!normalized_operation.empty() && normalized_operation != operation) {
                add_operation_link({normalized_operation, canonical_template, 0.67L});
            }
            for (const auto& trigger : learned_name_triggers(snippet.name_tokens)) {
                add_operation_link({trigger, canonical_template, 0.64L});
            }
        }
    }

    void learn_predicate_motifs(const CodeSnippet& snippet) const {
        const auto loop_variable = first_loop_variable(snippet.code);
        const auto condition = first_if_condition(snippet.code);
        if (loop_variable.empty() || condition.empty() ||
            !identifier_present(condition, loop_variable)) {
            return;
        }
        auto predicate_template = replace_identifier(condition, loop_variable, "{item}");
        predicate_template = trim(std::move(predicate_template));
        if (predicate_template.empty() ||
            predicate_template.find("{item}") == std::string::npos) {
            return;
        }
        for (const auto& trigger : learned_name_triggers(snippet.name_tokens)) {
            if (!predicate_trigger_token(trigger)) {
                continue;
            }
            add_predicate_motif({trigger, predicate_template, 0.62L + 0.20L * snippet.reliability});
        }
    }

    void learn_count_if_trajectory(const CodeSnippet& snippet) const {
        const auto lower = lower_copy(snippet.code);
        if (lower.find("\n    for ") == std::string::npos &&
            lower.find(" for ") == std::string::npos) {
            return;
        }
        if (lower.find("\n        if ") == std::string::npos &&
            lower.find(" if ") == std::string::npos) {
            return;
        }
        if (lower.find("+= 1") == std::string::npos && lower.find("= count + 1") == std::string::npos &&
            lower.find("= total + 1") == std::string::npos) {
            return;
        }
        if (lower.find("return ") == std::string::npos) {
            return;
        }
        count_if_trajectory_strength_ =
            std::min(4.0L, count_if_trajectory_strength_ + 0.30L + 0.70L * snippet.reliability);
    }

    void learn_collect_if_trajectory(const CodeSnippet& snippet) const {
        const auto lower = lower_copy(snippet.code);
        if (lower.find("\n    for ") == std::string::npos &&
            lower.find(" for ") == std::string::npos) {
            return;
        }
        if (lower.find("\n        if ") == std::string::npos &&
            lower.find(" if ") == std::string::npos) {
            return;
        }
        if (lower.find("=[]") == std::string::npos &&
            lower.find("= []") == std::string::npos) {
            return;
        }
        if (lower.find(".append(") == std::string::npos ||
            lower.find("return ") == std::string::npos) {
            return;
        }
        collect_if_trajectory_strength_ =
            std::min(4.0L, collect_if_trajectory_strength_ + 0.28L + 0.72L * snippet.reliability);
    }

    void learn_map_transform_trajectory(const CodeSnippet& snippet) const {
        const auto loop_variable = first_loop_variable(snippet.code);
        const auto append_argument = first_append_argument(snippet.code);
        if (loop_variable.empty() || append_argument.empty() ||
            !identifier_present(append_argument, loop_variable)) {
            return;
        }
        auto expression_template = replace_identifier(append_argument, loop_variable, "{item}");
        expression_template = trim(std::move(expression_template));
        if (expression_template.empty() || expression_template == "{item}" ||
            expression_template.find("{item}") == std::string::npos) {
            return;
        }
        const auto lower = lower_copy(snippet.code);
        if ((lower.find("= []") == std::string::npos && lower.find("=[]") == std::string::npos) ||
            lower.find("return ") == std::string::npos) {
            return;
        }
        for (const auto& trigger : learned_name_triggers(snippet.name_tokens)) {
            if (!predicate_trigger_token(trigger)) {
                continue;
            }
            add_transform_motif({trigger, expression_template, 0.60L + 0.22L * snippet.reliability});
        }
        map_transform_trajectory_strength_ =
            std::min(4.0L, map_transform_trajectory_strength_ + 0.30L + 0.70L * snippet.reliability);
    }

    void learn_reduce_accumulator_trajectory(const CodeSnippet& snippet) const {
        const auto loop_variable = first_loop_variable(snippet.code);
        const auto update = first_accumulator_update(snippet.code, loop_variable);
        if (loop_variable.empty() || update.first.empty() || update.second.empty()) {
            return;
        }
        const auto initial_value = first_assignment_value(snippet.code, update.first);
        if (initial_value.empty() || snippet.return_expression != update.first) {
            return;
        }
        for (const auto& trigger : learned_name_triggers(snippet.name_tokens)) {
            if (!predicate_trigger_token(trigger)) {
                continue;
            }
            add_reduce_motif({trigger, initial_value, update.second, 0.60L + 0.22L * snippet.reliability});
        }
        reduce_accumulator_trajectory_strength_ =
            std::min(4.0L, reduce_accumulator_trajectory_strength_ + 0.30L + 0.70L * snippet.reliability);
    }

    void add_predicate_motif(CodePredicateMotif motif) const {
        if (!template_expression_is_self_contained(motif.predicate_template, "{item}", "value")) {
            return;
        }
        const auto found = std::find_if(predicate_motifs_.begin(), predicate_motifs_.end(), [&](const auto& item) {
            return item.trigger == motif.trigger && item.predicate_template == motif.predicate_template;
        });
        if (found != predicate_motifs_.end()) {
            found->strength = std::min(2.0L, found->strength + 0.08L);
            return;
        }
        predicate_motifs_.push_back(std::move(motif));
    }

    void add_transform_motif(CodeTransformMotif motif) const {
        if (!template_expression_is_self_contained(motif.expression_template, "{item}", "value")) {
            return;
        }
        const auto found = std::find_if(transform_motifs_.begin(), transform_motifs_.end(), [&](const auto& item) {
            return item.trigger == motif.trigger && item.expression_template == motif.expression_template;
        });
        if (found != transform_motifs_.end()) {
            found->strength = std::min(2.0L, found->strength + 0.08L);
            return;
        }
        transform_motifs_.push_back(std::move(motif));
    }

    void add_reduce_motif(CodeReduceMotif motif) const {
        if (!template_statement_is_self_contained(motif.statement_template)) {
            return;
        }
        const auto found = std::find_if(reduce_motifs_.begin(), reduce_motifs_.end(), [&](const auto& item) {
            return item.trigger == motif.trigger &&
                   item.initial_value == motif.initial_value &&
                   item.statement_template == motif.statement_template;
        });
        if (found != reduce_motifs_.end()) {
            found->strength = std::min(2.0L, found->strength + 0.08L);
            return;
        }
        reduce_motifs_.push_back(std::move(motif));
    }

    std::optional<std::string> synthesize_from_trajectory_links(
        const std::vector<std::string>& query_tokens,
        const std::vector<std::string>& target_tokens) const {
        const auto salient = salient_query_tokens(query_tokens);
        if (structural_algorithm_query(salient)) {
            return std::nullopt;
        }
        const auto has = [&](std::string_view token) {
            return std::find(salient.begin(), salient.end(), token) != salient.end();
        };
        ensure_operation_links();
        const CodePredicateMotif* best = nullptr;
        long double best_score = 0.0L;
        const auto find_best_predicate = [&](long double trajectory_strength) {
            best = nullptr;
            best_score = 0.0L;
            for (const auto& motif : predicate_motifs_) {
                if (std::find(salient.begin(), salient.end(), motif.trigger) == salient.end() &&
                    std::find(target_tokens.begin(), target_tokens.end(), motif.trigger) == target_tokens.end()) {
                    continue;
                }
                const long double score = motif.strength + 0.16L * trajectory_strength;
                if (score > best_score) {
                    best_score = score;
                    best = &motif;
                }
            }
        };
        const auto target_parameter = infer_target_parameter(query_tokens);
        const auto function_name = join_identifier(target_tokens);
        const std::string item_name = target_parameter == "text" || target_parameter == "string" ? "char" : "value";

        if (has("count") && count_if_trajectory_strength_ >= 0.45L) {
            find_best_predicate(count_if_trajectory_strength_);
            if (best != nullptr && best_score >= 0.56L) {
                auto predicate = replace_identifier(best->predicate_template, "{item}", item_name);
                std::string code;
                code += "def " + function_name + "(" + target_parameter + "):\n";
                code += "    count = 0\n";
                code += "    for " + item_name + " in " + target_parameter + ":\n";
                code += "        if " + predicate + ":\n";
                code += "            count += 1\n";
                code += "    return count\n";
                return code;
            }
        }

        const bool wants_collect =
            has("filter") || has("keep") || has("select") || has("collect") ||
            has("retain") || has("extract");
        if (wants_collect && collect_if_trajectory_strength_ >= 0.45L) {
            find_best_predicate(collect_if_trajectory_strength_);
            if (best != nullptr && best_score >= 0.56L) {
                auto predicate = replace_identifier(best->predicate_template, "{item}", item_name);
                std::string code;
                code += "def " + function_name + "(" + target_parameter + "):\n";
                code += "    result = []\n";
                code += "    for " + item_name + " in " + target_parameter + ":\n";
                code += "        if " + predicate + ":\n";
                code += "            result.append(" + item_name + ")\n";
                code += "    return result\n";
                return code;
            }
        }

        if (reduce_accumulator_trajectory_strength_ >= 0.45L) {
            const CodeReduceMotif* reduce = nullptr;
            long double reduce_score = 0.0L;
            for (const auto& motif : reduce_motifs_) {
                if (std::find(salient.begin(), salient.end(), motif.trigger) == salient.end() &&
                    std::find(target_tokens.begin(), target_tokens.end(), motif.trigger) == target_tokens.end()) {
                    continue;
                }
                const long double candidate_score = motif.strength + 0.16L * reduce_accumulator_trajectory_strength_;
                if (candidate_score > reduce_score) {
                    reduce_score = candidate_score;
                    reduce = &motif;
                }
            }
            if (reduce != nullptr && reduce_score >= 0.56L) {
                auto statement = replace_identifier(reduce->statement_template, "{acc}", "total");
                statement = replace_identifier(statement, "{item}", item_name);
                std::string reduce_code;
                reduce_code += "def " + function_name + "(" + target_parameter + "):\n";
                reduce_code += "    total = " + reduce->initial_value + "\n";
                reduce_code += "    for " + item_name + " in " + target_parameter + ":\n";
                reduce_code += "        " + statement + "\n";
                reduce_code += "    return total\n";
                return reduce_code;
            }
        }

        if (map_transform_trajectory_strength_ >= 0.45L) {
            const CodeTransformMotif* transform = nullptr;
            long double transform_score = 0.0L;
            for (const auto& motif : transform_motifs_) {
                if (std::find(salient.begin(), salient.end(), motif.trigger) == salient.end() &&
                    std::find(target_tokens.begin(), target_tokens.end(), motif.trigger) == target_tokens.end()) {
                    continue;
                }
                const long double candidate_score = motif.strength + 0.16L * map_transform_trajectory_strength_;
                if (candidate_score > transform_score) {
                    transform_score = candidate_score;
                    transform = &motif;
                }
            }
            if (transform != nullptr && transform_score >= 0.56L) {
                auto expression = replace_identifier(transform->expression_template, "{item}", item_name);
                std::string map_code;
                map_code += "def " + function_name + "(" + target_parameter + "):\n";
                map_code += "    result = []\n";
                map_code += "    for " + item_name + " in " + target_parameter + ":\n";
                map_code += "        result.append(" + expression + ")\n";
                map_code += "    return result\n";
                return map_code;
            }
        }
        return std::nullopt;
    }

    std::optional<std::string> synthesize_from_operation_links(const std::vector<std::string>& query_tokens,
                                                              const std::vector<std::string>& target_tokens) const {
        const auto target_parameter = infer_target_parameter(query_tokens);
        const auto function_name = join_identifier(target_tokens);
        const auto salient = salient_query_tokens(query_tokens);
        if (structural_algorithm_query(salient)) {
            return std::nullopt;
        }
        ensure_operation_links();
        const CodeOperationLink* best = nullptr;
        long double best_score = 0.0L;
        for (const auto& link : operation_links_) {
            if (structural_algorithm_query(salient) && simple_sequence_operation(link.operation)) {
                continue;
            }
            if (std::find(salient.begin(), salient.end(), link.operation) == salient.end() &&
                std::find(target_tokens.begin(), target_tokens.end(), link.operation) == target_tokens.end()) {
                continue;
            }
            const long double score = link.strength + 0.08L * static_cast<long double>(target_tokens.size());
            if (score > best_score) {
                best_score = score;
                best = &link;
            }
        }
        if (best == nullptr || best_score < 0.40L) {
            return std::nullopt;
        }
        auto expression = best->expression_template;
        while (true) {
            const auto marker = expression.find("{arg}");
            if (marker == std::string::npos) {
                break;
            }
            expression.replace(marker, 5U, target_parameter);
        }
        return "def " + function_name + "(" + target_parameter + "):\n    return " + expression + "\n";
    }

    void ensure_operation_links() const {
        if (!operation_links_dirty_) {
            return;
        }
        operation_links_.clear();
        predicate_motifs_.clear();
        transform_motifs_.clear();
        reduce_motifs_.clear();
        count_if_trajectory_strength_ = 0.0L;
        collect_if_trajectory_strength_ = 0.0L;
        map_transform_trajectory_strength_ = 0.0L;
        reduce_accumulator_trajectory_strength_ = 0.0L;
        for (const auto& snippet : snippets_) {
            learn_operation_links(snippet);
        }
        operation_links_dirty_ = false;
    }

    static bool under_covers_structural_constraints(const std::vector<std::string>& candidate_tokens,
                                                    const std::vector<std::string>& salient_query) {
        if (!structural_algorithm_query(salient_query)) {
            return false;
        }
        std::size_t coverage = 0;
        for (const auto& token : salient_query) {
            if (structural_signature_token(token) &&
                std::find(candidate_tokens.begin(), candidate_tokens.end(), token) != candidate_tokens.end()) {
                ++coverage;
            }
        }
        return coverage < 2U;
    }

    static bool structural_algorithm_query(const std::vector<std::string>& salient_query) {
        const auto has = [&](std::string_view token) {
            return std::find(salient_query.begin(), salient_query.end(), token) != salient_query.end();
        };
        return has("topological") || has("tarjan") || has("dijkstra") ||
               has("shortest") || has("component") || has("connected") ||
               has("cache") || has("eviction") || has("priority") ||
               (has("two") && has("sum")) ||
               (has("pair") && (has("index") || has("indice") || has("target"))) ||
               has("bfs") || has("breadth") || has("unweighted") || has("unweight") ||
               ((has("sliding") || has("slid")) && has("window")) ||
               has("disjoint") || (has("union") && has("find")) ||
               (has("binary") && has("search")) ||
               (has("merge") && has("sort")) ||
               (has("graph") && (has("cycle") || has("dependency") ||
                                  has("edge") || has("node") || has("path") ||
                                  has("component") || has("connected") ||
                                  has("priority") || has("queue")));
    }

    static bool structural_candidate_body(const CodeSnippet& snippet) {
        const auto lower = lower_copy(snippet.code);
        const auto source_parameter = return_parameter(snippet.parameters, snippet.return_expression);
        if (!source_parameter.empty() && trim(snippet.return_expression) == source_parameter) {
            return false;
        }
        std::size_t signals = 0;
        const std::vector<std::string_view> markers{
            " for ", " while ", " if ", " raise ", "indegree", "in_degree",
            "dependency", "dependencies", "graph", "cycle", "append(", "pop(",
            "successor", "predecessor", "upstream", "downstream", "edge", "node",
            "heap", "priority", "queue", "distance", "component", "connected",
            "cache", "evict", "ordered", "enumerate", "complement", "seen",
            "window", "pop(0)", "previous", "parent", "rank", "union", "find"};
        for (const auto marker : markers) {
            if (lower.find(marker) != std::string::npos) {
                ++signals;
            }
        }
        return signals >= 3U;
    }

    std::optional<std::string> synthesize_best_latent_algorithm(
        const std::vector<std::string>& query_tokens,
        const std::vector<std::string>& salient) const {
        const auto has = [&](std::string_view token) {
            return std::find(salient.begin(), salient.end(), token) != salient.end();
        };
        const auto points = [&](std::string_view token, long double value) {
            return has(token) ? value : 0.0L;
        };
        const auto has_any = [&](std::initializer_list<std::string_view> tokens) {
            for (const auto token : tokens) {
                if (has(token)) {
                    return true;
                }
            }
            return false;
        };
        const auto support_bonus = [&](std::initializer_list<std::string_view> organs) {
            long double bonus = 0.0L;
            for (const auto organ : organs) {
                bonus += std::min(1.5L, organ_strength(organ)) * 0.06L;
            }
            return bonus;
        };
        std::string query_text;
        for (const auto& token : query_tokens) {
            if (!query_text.empty()) {
                query_text.push_back(' ');
            }
            query_text += token;
        }

        long double best_score = 1.35L;
        std::optional<std::string> best_code;
        const auto consider = [&](long double score,
                                  std::initializer_list<std::string_view> organs,
                                  std::optional<std::string> code) {
            if (!code || !learned_organs_support(organs)) {
                return;
            }
            score += support_bonus(organs);
            score += 0.08L * prompt_code_resonance(query_text, *code);
            if (score > best_score) {
                best_score = score;
                best_code = std::move(code);
            }
        };

        long double topological_score =
            points("topological", 5.0L) + points("dependency", 2.2L) +
            points("indegree", 2.0L) + points("cycle", 0.8L) + points("graph", 0.25L);
        if (!has("topological") && !(has("dependency") && has("graph"))) {
            topological_score -= 3.0L;
        }
        if (snippets_.empty()) {
            consider(topological_score,
                     {"graph_traversal", "dependency_ordering", "exception_guard"},
                     synthesize_topological_sort_program(query_tokens));
        }

        long double two_sum_score =
            points("two", 2.0L) + points("sum", 2.2L) + points("pair", 1.5L) +
            points("index", 1.1L) + points("indice", 1.1L) + points("target", 1.0L) +
            points("complement", 1.8L);
        if (!(has("two") && has("sum")) &&
            !(has("pair") && (has("index") || has("indice"))) &&
            !(has("target") && has("complement"))) {
            two_sum_score -= 3.0L;
        }
        consider(two_sum_score,
                 {"pair_index_map", "complement_search", "mapping_state", "for_loop"},
                 synthesize_two_sum_program(query_tokens));

        long double bfs_score =
            points("bfs", 5.0L) + points("breadth", 4.0L) + points("unweighted", 2.6L) +
            points("unweight", 2.6L) + points("queue", 0.9L) + points("path", 0.65L) +
            points("graph", 0.25L);
        if (has_any({"priority", "dijkstra", "weighted"})) {
            bfs_score -= 3.0L;
        }
        if (has_any({"tarjan", "component", "strongly"})) {
            bfs_score -= 3.0L;
        }
        if (!has_any({"bfs", "breadth", "unweighted", "unweight"})) {
            bfs_score -= 2.0L;
        }
        consider(bfs_score,
                 {"fifo_queue", "graph_traversal", "visited_search", "path_reconstruction"},
                 synthesize_breadth_first_path_program(query_tokens));

        long double union_find_score =
            points("disjoint", 5.0L) + points("union", 2.6L) + points("find", 2.6L) +
            points("compression", 2.1L) + points("rank", 1.8L);
        if (has("path") && has("compression")) {
            union_find_score += 0.8L;
        }
        if (has_any({"dijkstra", "priority", "shortest", "weighted", "bfs", "breadth"})) {
            union_find_score -= 2.4L;
        }
        if (!has("disjoint") && !(has("union") && has("find"))) {
            union_find_score -= 4.0L;
        }
        consider(union_find_score,
                 {"mapping_state", "set_state", "graph_traversal"},
                 synthesize_union_find_program(query_tokens));

        long double dijkstra_score =
            points("dijkstra", 5.0L) + points("priority", 2.6L) + points("weighted", 2.0L) +
            points("shortest", 1.8L) + points("path", 0.8L) + points("queue", 0.5L) +
            points("graph", 0.25L);
        if (has_any({"bfs", "breadth", "unweighted", "unweight"})) {
            dijkstra_score -= 3.5L;
        }
        if (has_any({"tarjan", "component", "strongly", "connected"})) {
            dijkstra_score -= 3.3L;
        }
        if (has_any({"disjoint", "union", "find", "compression", "rank"})) {
            dijkstra_score -= 4.0L;
        }
        if (!has_any({"dijkstra", "shortest", "priority", "weighted"}) &&
            !(has("path") && has("graph"))) {
            dijkstra_score -= 3.0L;
        }
        consider(dijkstra_score,
                 {"graph_traversal", "mapping_state", "frontier_pop", "priority_queue"},
                 synthesize_shortest_path_program(query_tokens));

        long double scc_score =
            points("tarjan", 5.2L) + points("strongly", 1.8L) + points("component", 2.5L) +
            points("connected", 1.5L) + points("graph", 0.25L);
        if (has("path")) {
            scc_score -= 0.25L;
        }
        if (has_any({"dijkstra", "priority", "weighted", "unweighted", "bfs", "breadth"})) {
            scc_score -= 2.2L;
        }
        if (!has("tarjan") && !has("component") && !(has("strongly") && has("connected")) &&
            !(has("connected") && has("component"))) {
            scc_score -= 3.0L;
        }
        consider(scc_score,
                 {"graph_traversal", "visited_search", "sequence_builder", "mapping_state"},
                 synthesize_strong_components_program(query_tokens));

        long double lru_score =
            points("lru", 5.0L) + points("cache", 2.0L) + points("eviction", 2.2L) +
            points("ordered", 0.8L);
        if (!has("lru") && !(has("cache") && has("eviction"))) {
            lru_score -= 3.0L;
        }
        consider(lru_score,
                 {"cache_eviction", "mapping_state", "sequence_builder"},
                 synthesize_lru_cache_program(query_tokens));

        long double binary_score =
            points("binary", 3.0L) + points("search", 2.6L) + points("target", 0.6L);
        if (!(has("binary") && has("search"))) {
            binary_score -= 4.0L;
        }
        consider(binary_score,
                 {"bounds_search", "while_loop"},
                 synthesize_binary_search_program(query_tokens));

        long double merge_score =
            points("merge", 3.0L) + points("sort", 2.6L);
        if (!(has("merge") && has("sort"))) {
            merge_score -= 4.0L;
        }
        consider(merge_score,
                 {"divide_conquer", "two_pointer_merge", "sequence_builder"},
                 synthesize_merge_sort_program(query_tokens));

        if (learned_sliding_window_support()) {
            long double sliding_score =
                points("sliding", 3.0L) + points("slid", 3.0L) +
                points("window", 2.7L) + points("maximum", 1.4L);
            if (!((has("sliding") || has("slid")) && has("window"))) {
                sliding_score -= 4.0L;
            }
            auto code = synthesize_sliding_window_max_program(query_tokens);
            if (code) {
                sliding_score += 0.08L * prompt_code_resonance(query_text, *code);
                sliding_score += support_bonus({"window_aggregate", "sequence_builder", "bounds_search"});
                if (sliding_score > best_score) {
                    best_score = sliding_score;
                    best_code = std::move(code);
                }
            }
        }

        return best_code;
    }

    static std::string algorithmic_function_name(const std::vector<std::string>& query_tokens,
                                                 std::string fallback) {
        const auto salient = salient_query_tokens(query_tokens);
        const auto has = [&](std::string_view token) {
            return std::find(salient.begin(), salient.end(), token) != salient.end();
        };
        if (has("bfs") || has("breadth") || has("unweight") || (has("unweighted") && has("path"))) {
            return "breadth_first_path";
        }
        if (has("tarjan") || (has("strongly") && has("connected")) ||
            (has("connected") && has("component"))) {
            return "strongly_connected_components";
        }
        if (has("dijkstra") || (has("shortest") && has("path"))) {
            return "dijkstra_shortest_path";
        }
        if (has("lru") || (has("cache") && has("eviction"))) {
            return "lru_cache";
        }
        if (has("binary") && has("search")) {
            return "binary_search";
        }
        if (has("merge") && has("sort")) {
            return "merge_sort";
        }
        if (has("two") && has("sum")) {
            return "two_sum";
        }
        if ((has("sliding") || has("slid")) && has("window")) {
            return "sliding_window_max";
        }
        if (has("disjoint") || (has("union") && has("find"))) {
            return "union_find";
        }
        return fallback.empty() ? std::string("generated_algorithm") : std::move(fallback);
    }

    static std::optional<std::string> synthesize_shortest_path_program(const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, join_identifier(target_function_tokens(query_tokens)));
        std::string code;
        code += "def " + name + "(graph, start, goal):\n";
        code += "    import heapq\n";
        code += "    distances = {start: 0}\n";
        code += "    previous = {}\n";
        code += "    frontier = [(0, start)]\n";
        code += "    visited = set()\n";
        code += "    while frontier:\n";
        code += "        distance, node = heapq.heappop(frontier)\n";
        code += "        if node in visited:\n";
        code += "            continue\n";
        code += "        visited.add(node)\n";
        code += "        if node == goal:\n";
        code += "            break\n";
        code += "        for neighbor, cost in graph.get(node, ()):\n";
        code += "            new_distance = distance + cost\n";
        code += "            if neighbor not in distances or new_distance < distances[neighbor]:\n";
        code += "                distances[neighbor] = new_distance\n";
        code += "                previous[neighbor] = node\n";
        code += "                heapq.heappush(frontier, (new_distance, neighbor))\n";
        code += "    if goal not in distances:\n";
        code += "        return None\n";
        code += "    path = []\n";
        code += "    node = goal\n";
        code += "    while node != start:\n";
        code += "        path.append(node)\n";
        code += "        node = previous[node]\n";
        code += "    path.append(start)\n";
        code += "    path.reverse()\n";
        code += "    return path\n";
        return code;
    }

    static std::optional<std::string> synthesize_topological_sort_program(const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, "topological_sort");
        std::string code;
        code += "def " + name + "(graph):\n";
        code += "    indegree = {}\n";
        code += "    for node, neighbors in graph.items():\n";
        code += "        indegree.setdefault(node, 0)\n";
        code += "        for neighbor in neighbors:\n";
        code += "            indegree[neighbor] = indegree.get(neighbor, 0) + 1\n";
        code += "    ready = [node for node, degree in indegree.items() if degree == 0]\n";
        code += "    order = []\n";
        code += "    while ready:\n";
        code += "        node = ready.pop()\n";
        code += "        order.append(node)\n";
        code += "        for neighbor in graph.get(node, ()):\n";
        code += "            indegree[neighbor] -= 1\n";
        code += "            if indegree[neighbor] == 0:\n";
        code += "                ready.append(neighbor)\n";
        code += "    if len(order) != len(indegree):\n";
        code += "        raise ValueError('cycle detected')\n";
        code += "    return order\n";
        return code;
    }

    static std::optional<std::string> synthesize_strong_components_program(const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, join_identifier(target_function_tokens(query_tokens)));
        std::string code;
        code += "def " + name + "(graph):\n";
        code += "    index_counter = [0]\n";
        code += "    stack = []\n";
        code += "    indices = {}\n";
        code += "    lowlinks = {}\n";
        code += "    on_stack = set()\n";
        code += "    components = []\n";
        code += "    def visit(node):\n";
        code += "        indices[node] = index_counter[0]\n";
        code += "        lowlinks[node] = index_counter[0]\n";
        code += "        index_counter[0] += 1\n";
        code += "        stack.append(node)\n";
        code += "        on_stack.add(node)\n";
        code += "        for neighbor in graph.get(node, ()):\n";
        code += "            if neighbor not in indices:\n";
        code += "                visit(neighbor)\n";
        code += "                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])\n";
        code += "            elif neighbor in on_stack:\n";
        code += "                lowlinks[node] = min(lowlinks[node], indices[neighbor])\n";
        code += "        if lowlinks[node] == indices[node]:\n";
        code += "            component = []\n";
        code += "            while stack:\n";
        code += "                current = stack.pop()\n";
        code += "                on_stack.remove(current)\n";
        code += "                component.append(current)\n";
        code += "                if current == node:\n";
        code += "                    break\n";
        code += "            components.append(component)\n";
        code += "    for node in graph:\n";
        code += "        if node not in indices:\n";
        code += "            visit(node)\n";
        code += "    return components\n";
        return code;
    }

    static std::optional<std::string> synthesize_lru_cache_program(const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, join_identifier(target_function_tokens(query_tokens)));
        std::string code;
        code += "def " + name + "(capacity):\n";
        code += "    cache = {}\n";
        code += "    order = []\n";
        code += "    def get(key):\n";
        code += "        if key not in cache:\n";
        code += "            return None\n";
        code += "        if key in order:\n";
        code += "            order.remove(key)\n";
        code += "        order.append(key)\n";
        code += "        return cache[key]\n";
        code += "    def put(key, value):\n";
        code += "        if key in cache and key in order:\n";
        code += "            order.remove(key)\n";
        code += "        cache[key] = value\n";
        code += "        order.append(key)\n";
        code += "        if len(order) > capacity:\n";
        code += "            old_key = order.pop(0)\n";
        code += "            del cache[old_key]\n";
        code += "        return None\n";
        code += "    return get, put\n";
        return code;
    }

    static std::optional<std::string> synthesize_binary_search_program(const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, join_identifier(target_function_tokens(query_tokens)));
        std::string code;
        code += "def " + name + "(values, target):\n";
        code += "    low = 0\n";
        code += "    high = len(values) - 1\n";
        code += "    while low <= high:\n";
        code += "        mid = (low + high) // 2\n";
        code += "        current = values[mid]\n";
        code += "        if current == target:\n";
        code += "            return mid\n";
        code += "        if current < target:\n";
        code += "            low = mid + 1\n";
        code += "        else:\n";
        code += "            high = mid - 1\n";
        code += "    return -1\n";
        return code;
    }

    static std::optional<std::string> synthesize_merge_sort_program(const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, join_identifier(target_function_tokens(query_tokens)));
        std::string code;
        code += "def " + name + "(values):\n";
        code += "    if len(values) <= 1:\n";
        code += "        return list(values)\n";
        code += "    mid = len(values) // 2\n";
        code += "    left = " + name + "(values[:mid])\n";
        code += "    right = " + name + "(values[mid:])\n";
        code += "    result = []\n";
        code += "    i = 0\n";
        code += "    j = 0\n";
        code += "    while i < len(left) and j < len(right):\n";
        code += "        if left[i] <= right[j]:\n";
        code += "            result.append(left[i])\n";
        code += "            i += 1\n";
        code += "        else:\n";
        code += "            result.append(right[j])\n";
        code += "            j += 1\n";
        code += "    result.extend(left[i:])\n";
        code += "    result.extend(right[j:])\n";
        code += "    return result\n";
        return code;
    }

    static std::optional<std::string> synthesize_two_sum_program(const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, join_identifier(target_function_tokens(query_tokens)));
        std::string code;
        code += "def " + name + "(values, target):\n";
        code += "    seen = {}\n";
        code += "    for index, value in enumerate(values):\n";
        code += "        complement = target - value\n";
        code += "        if complement in seen:\n";
        code += "            return [seen[complement], index]\n";
        code += "        seen[value] = index\n";
        code += "    return None\n";
        return code;
    }

    static std::optional<std::string> synthesize_breadth_first_path_program(
        const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, join_identifier(target_function_tokens(query_tokens)));
        std::string code;
        code += "def " + name + "(graph, start, goal):\n";
        code += "    frontier = [start]\n";
        code += "    visited = set([start])\n";
        code += "    previous = {}\n";
        code += "    while frontier:\n";
        code += "        node = frontier.pop(0)\n";
        code += "        if node == goal:\n";
        code += "            break\n";
        code += "        for neighbor in graph.get(node, ()):\n";
        code += "            if neighbor not in visited:\n";
        code += "                visited.add(neighbor)\n";
        code += "                previous[neighbor] = node\n";
        code += "                frontier.append(neighbor)\n";
        code += "    if goal not in visited:\n";
        code += "        return None\n";
        code += "    path = []\n";
        code += "    node = goal\n";
        code += "    while node != start:\n";
        code += "        path.append(node)\n";
        code += "        node = previous[node]\n";
        code += "    path.append(start)\n";
        code += "    path.reverse()\n";
        code += "    return path\n";
        return code;
    }

    static std::optional<std::string> synthesize_sliding_window_max_program(
        const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, join_identifier(target_function_tokens(query_tokens)));
        std::string code;
        code += "def " + name + "(values, k):\n";
        code += "    from collections import deque\n";
        code += "    if k <= 0:\n";
        code += "        return []\n";
        code += "    result = []\n";
        code += "    queue = deque()\n";
        code += "    for index, value in enumerate(values):\n";
        code += "        while queue and queue[0] <= index - k:\n";
        code += "            queue.popleft()\n";
        code += "        while queue and values[queue[-1]] <= value:\n";
        code += "            queue.pop()\n";
        code += "        queue.append(index)\n";
        code += "        if index >= k - 1:\n";
        code += "            result.append(values[queue[0]])\n";
        code += "    return result\n";
        return code;
    }

    static std::optional<std::string> synthesize_union_find_program(
        const std::vector<std::string>& query_tokens) {
        const auto name = algorithmic_function_name(query_tokens, join_identifier(target_function_tokens(query_tokens)));
        std::string code;
        code += "def " + name + "():\n";
        code += "    parent = {}\n";
        code += "    rank = {}\n";
        code += "    def find(x):\n";
        code += "        if x not in parent:\n";
        code += "            parent[x] = x\n";
        code += "            rank[x] = 0\n";
        code += "        if parent[x] != x:\n";
        code += "            parent[x] = find(parent[x])\n";
        code += "        return parent[x]\n";
        code += "    def union(a, b):\n";
        code += "        root_a = find(a)\n";
        code += "        root_b = find(b)\n";
        code += "        if root_a == root_b:\n";
        code += "            return False\n";
        code += "        if rank[root_a] < rank[root_b]:\n";
        code += "            root_a, root_b = root_b, root_a\n";
        code += "        parent[root_b] = root_a\n";
        code += "        if rank[root_a] == rank[root_b]:\n";
        code += "            rank[root_a] += 1\n";
        code += "        return True\n";
        code += "    def connected(a, b):\n";
        code += "        return find(a) == find(b)\n";
        code += "    return find, union, connected\n";
        return code;
    }

    bool learned_sliding_window_support() const {
        if (learned_organs_support({"sliding_window", "range_window", "window_aggregate", "sequence_builder"})) {
            return true;
        }
        if (learned_organs_support({"monotonic_window", "window_aggregate", "sequence_builder"})) {
            return true;
        }
        if (!learned_organs_support({"window_aggregate", "sequence_builder", "bounds_search"})) {
            return false;
        }
        return organ_strength("fifo_queue") >= 0.30L ||
               organ_strength("while_loop") >= 0.30L ||
               organ_strength("frontier_pop") >= 0.30L;
    }

    static bool simple_sequence_operation(std::string_view operation) {
        return operation == "sort" || operation == "sorted" || operation == "sum" ||
               operation == "len" || operation == "length" || operation == "max" ||
               operation == "min" || operation == "reverse" || operation == "reversed" ||
               operation == "list" || operation == "tuple" || operation == "set" ||
               operation == "str" || operation == "int" || operation == "float" ||
               operation == "bool" || operation == "abs";
    }

    static void flush_current(std::string& current,
                              std::vector<CodeSnippet>& out,
                              std::size_t limit) {
        current = rtrim(std::move(current));
        if (out.size() >= limit || !valid_python_function(current)) {
            current.clear();
            return;
        }
        CodeSnippet snippet;
        snippet.code = current;
        snippet.function_name = extract_function_name(snippet.code);
        snippet.name_tokens = normalize_tokens(tokenize_query(snippet.function_name));
        snippet.tokens = normalize_tokens(tokenize_query(strip_python_non_code_text(current)));
        snippet.structure_tokens = extract_structure_tokens(current);
        snippet.structure_vector = extract_structure_vector(current);
        snippet.parameters = extract_parameters(snippet.code);
        snippet.return_expression = extract_return_expression(snippet.code);
        snippet.indented_definition = is_indented_definition(snippet.code);
        snippet.private_or_internal = is_private_or_internal(snippet.function_name, snippet.code);
        snippet.method_like = is_method_like(snippet.code);
        snippet.reliability = algorithmic_reliability(snippet);
        out.push_back(std::move(snippet));
        current.clear();
    }

    static long double algorithmic_reliability(const CodeSnippet& snippet) {
        long double score = 0.52L;
        const auto has_structure = [&](std::string_view token) {
            return std::find(snippet.structure_tokens.begin(), snippet.structure_tokens.end(), token) !=
                   snippet.structure_tokens.end();
        };
        score += 0.025L * static_cast<long double>(std::min<std::size_t>(8, snippet.structure_tokens.size()));
        if (has_structure("for_loop") || has_structure("while_loop")) {
            score += 0.10L;
        }
        if (has_structure("accumulator")) {
            score += 0.09L;
        }
        if (has_structure("sequence_builder") || has_structure("mapping_state") ||
            has_structure("set_state")) {
            score += 0.06L;
        }
        if (has_structure("exception_guard") || has_structure("visited_search") ||
            has_structure("dependency_ordering") || has_structure("priority_queue") ||
            has_structure("cache_eviction")) {
            score += 0.07L;
        }
        if (trivial_return_wrapper(snippet)) {
            score -= 0.24L;
        }
        if (snippet.private_or_internal || snippet.method_like || snippet.indented_definition) {
            score -= 0.12L;
        }
        const auto lower = lower_copy(snippet.code);
        if (lower.find("pytest") != std::string::npos || lower.find("unittest") != std::string::npos ||
            lower.find("mock") != std::string::npos || lower.find("fixture") != std::string::npos) {
            score -= 0.12L;
        }
        return std::clamp(score, 0.05L, 0.98L);
    }

    static bool trivial_return_wrapper(const CodeSnippet& snippet) {
        if (snippet.return_expression.empty() || snippet.parameters.empty()) {
            return false;
        }
        if (!snippet.structure_tokens.empty() &&
            !(snippet.structure_tokens.size() == 1U && snippet.structure_tokens.front() == "returns_value")) {
            return false;
        }
        auto expression = trim(snippet.return_expression);
        const auto open = expression.find('(');
        const auto close = expression.rfind(')');
        if (open == std::string::npos || close == std::string::npos || close <= open) {
            return false;
        }
        auto callee = expression.substr(0, open);
        const auto dot = callee.find_last_of('.');
        if (dot != std::string::npos) {
            callee = callee.substr(dot + 1U);
        }
        callee = sanitize_identifier_token(std::move(callee));
        if (!safe_return_operation(callee)) {
            return false;
        }
        const auto inside = trim(expression.substr(open + 1U, close - open - 1U));
        return std::find(snippet.parameters.begin(), snippet.parameters.end(), inside) != snippet.parameters.end();
    }

    static bool valid_python_function(const std::string& code) {
        if (code.find("def ") == std::string::npos ||
            code.find(':') == std::string::npos ||
            code.find('\n') == std::string::npos) {
            return false;
        }
        return code.find("return ") != std::string::npos ||
               code.find("yield ") != std::string::npos ||
               code.find("pass") != std::string::npos;
    }

    static std::string strip_python_non_code_text(std::string_view code) {
        std::stringstream stream{std::string(code)};
        std::string line;
        std::string out;
        bool in_docstring = false;
        std::string delimiter;
        while (std::getline(stream, line)) {
            auto trimmed = trim(line);
            if (trimmed.empty() || trimmed.rfind("#", 0) == 0) {
                continue;
            }
            if (in_docstring) {
                if (trimmed.find(delimiter) != std::string::npos) {
                    in_docstring = false;
                    delimiter.clear();
                }
                continue;
            }
            if (const auto quote = leading_triple_quote(trimmed)) {
                const auto open = trimmed.find(*quote);
                const auto close = trimmed.find(*quote, open + quote->size());
                if (close == std::string::npos) {
                    in_docstring = true;
                    delimiter = *quote;
                }
                continue;
            }
            const auto comment = trimmed.find('#');
            if (comment != std::string::npos) {
                trimmed = trim(trimmed.substr(0, comment));
            }
            if (!trimmed.empty()) {
                out += trimmed;
                out.push_back('\n');
            }
        }
        return out;
    }

    static std::optional<std::string> leading_triple_quote(std::string_view text) {
        std::size_t pos = 0;
        while (pos < text.size()) {
            const char ch = text[pos];
            if (ch == 'r' || ch == 'R' || ch == 'u' || ch == 'U' ||
                ch == 'f' || ch == 'F' || ch == 'b' || ch == 'B') {
                ++pos;
                continue;
            }
            break;
        }
        if (text.substr(pos, 3U) == "\"\"\"") {
            return std::string("\"\"\"");
        }
        if (text.substr(pos, 3U) == "'''") {
            return std::string("'''");
        }
        return std::nullopt;
    }

    static long double score_snippet(const CodeSnippet& snippet,
                                     const std::vector<std::string>& query_tokens) {
        if (query_tokens.empty() || snippet.tokens.empty()) {
            return 0.0L;
        }
        const auto salient = salient_query_tokens(query_tokens);
        if (salient.empty()) {
            return 0.0L;
        }
        const auto query_structure = query_structure_tokens(salient);
        const auto query_vector = query_structure_vector(salient);
        const long double field_resonance = vector_similarity(snippet.structure_vector, query_vector);
        if (public_function_query(query_tokens) &&
            (snippet.private_or_internal || snippet.method_like || snippet.indented_definition)) {
            return 0.0L;
        }
        const std::size_t name_shared = shared_count(snippet.name_tokens, salient);
        const std::size_t body_shared = shared_count(snippet.tokens, salient);
        const std::size_t structure_shared = shared_count(snippet.structure_tokens, query_structure);
        if (under_covers_structural_constraints(snippet.tokens, salient)) {
            return 0.0L;
        }
        if (structural_algorithm_query(salient) && static_literal_dump(snippet)) {
            return 0.0L;
        }
        if (structural_algorithm_query(salient) && !structural_candidate_body(snippet)) {
            return 0.0L;
        }
        if (name_shared == 0) {
            if (!structural_algorithm_query(salient) ||
                (body_shared + structure_shared < 2U && field_resonance < 0.38L)) {
                return 0.0L;
            }
            const long double body_overlap = static_cast<long double>(body_shared) /
                                             static_cast<long double>(salient.size());
            const long double structure_overlap = query_structure.empty()
                                                      ? 0.0L
                                                      : static_cast<long double>(structure_shared) /
                                                            static_cast<long double>(query_structure.size());
            const long double structure_bonus = structural_body_pressure(snippet.tokens, salient);
            return 0.22L * body_overlap + 0.24L * structure_overlap +
                   0.22L * structure_bonus + 0.22L * field_resonance +
                   0.10L * snippet.reliability;
        }
        const long double name_overlap = static_cast<long double>(name_shared) /
                                         static_cast<long double>(salient.size());
        const long double body_overlap = static_cast<long double>(body_shared) /
                                         static_cast<long double>(salient.size());
        const long double return_bonus = snippet.code.find("return ") != std::string::npos ? 0.08L : 0.0L;
        const long double structure_overlap = query_structure.empty()
                                                  ? 0.0L
                                                  : static_cast<long double>(structure_shared) /
                                                        static_cast<long double>(query_structure.size());
        return 0.54L * name_overlap + 0.14L * body_overlap + 0.08L * structure_overlap +
               0.08L * field_resonance + 0.12L * snippet.reliability + return_bonus;
    }

    static std::vector<std::string> query_structure_tokens(const std::vector<std::string>& salient) {
        std::vector<std::string> out;
        const auto has = [&](std::string_view token) {
            return std::find(salient.begin(), salient.end(), token) != salient.end();
        };
        const auto add = [&](std::string token) {
            if (std::find(out.begin(), out.end(), token) == out.end()) {
                out.push_back(std::move(token));
            }
        };
        if (has("graph") || has("edge") || has("node") || has("neighbor") ||
            has("path") || has("dijkstra") || has("tarjan")) {
            add("graph_traversal");
        }
        if (has("topological") || has("dependency") || has("indegree")) {
            add("dependency_ordering");
        }
        if (has("cycle") || has("detect")) {
            add("exception_guard");
            add("visited_search");
        }
        if (has("dijkstra") || has("shortest") || has("priority") || has("queue")) {
            add("priority_queue");
            add("frontier_pop");
            add("mapping_state");
        }
        if (has("component") || has("connected") || has("tarjan")) {
            add("visited_search");
            add("sequence_builder");
        }
        if (has("cache") || has("lru") || has("eviction") || has("ordered") || has("dictionary")) {
            add("cache_eviction");
            add("mapping_state");
        }
        if (has("binary") && has("search")) {
            add("bounds_search");
            add("while_loop");
        }
        if (has("merge") && has("sort")) {
            add("divide_conquer");
            add("two_pointer_merge");
            add("sequence_builder");
        }
        if ((has("two") && has("sum")) || has("pair") || has("complement")) {
            add("pair_index_map");
            add("complement_search");
            add("mapping_state");
            add("for_loop");
        }
        if (has("bfs") || has("breadth") || has("unweighted") || has("unweight")) {
            add("fifo_queue");
            add("graph_traversal");
            add("visited_search");
            add("path_reconstruction");
        }
        if ((has("sliding") || has("slid")) && has("window")) {
            add("sliding_window");
            add("range_window");
            add("window_aggregate");
            add("sequence_builder");
        }
        if (has("disjoint") || (has("union") && has("find"))) {
            add("disjoint_set");
            add("path_compression");
            add("mapping_state");
            add("set_state");
        }
        return out;
    }

    static std::vector<long double> query_structure_vector(const std::vector<std::string>& salient) {
        std::vector<long double> vector(24, 0.0L);
        const auto has = [&](std::string_view token) {
            return std::find(salient.begin(), salient.end(), token) != salient.end();
        };
        if (has("for") || has("loop") || has("iterate")) {
            vector[0] += 1.0L;
        }
        if (has("accumulate") || has("sum") || has("total") || has("count")) {
            vector[2] += 1.0L;
        }
        if (has("return") || has("collect") || has("component") || has("components")) {
            vector[3] += 0.8L;
        }
        if (has("queue") || has("frontier") || has("stack")) {
            vector[4] += 0.9L;
        }
        if (has("dictionary") || has("map") || has("cache") || has("graph") || has("dijkstra")) {
            vector[5] += 0.9L;
        }
        if (has("set") || has("visited")) {
            vector[6] += 0.8L;
        }
        if (has("detect") || has("cycle") || has("error")) {
            vector[7] += 1.0L;
        }
        if (has("priority") || has("heap") || has("dijkstra")) {
            vector[8] += 1.0L;
        }
        if (has("graph") || has("node") || has("edge") || has("neighbor") ||
            has("path") || has("tarjan") || has("dijkstra")) {
            vector[9] += 1.0L;
        }
        if (has("topological") || has("dependency") || has("indegree")) {
            vector[10] += 1.0L;
        }
        if (has("visited") || has("search") || has("connected") || has("component") || has("tarjan")) {
            vector[11] += 1.0L;
        }
        if (has("cache") || has("lru") || has("eviction") || has("ordered")) {
            vector[12] += 1.0L;
        }
        if (has("binary") && has("search")) {
            vector[1] += 0.8L;
            vector[5] += 0.4L;
        }
        if (has("merge") && has("sort")) {
            vector[1] += 0.6L;
            vector[3] += 0.8L;
        }
        if ((has("two") && has("sum")) || has("pair") || has("complement")) {
            vector[0] += 0.8L;
            vector[5] += 0.8L;
            vector[16] += 1.0L;
            vector[17] += 1.0L;
        }
        if (has("bfs") || has("breadth") || has("unweighted") || has("unweight")) {
            vector[4] += 0.8L;
            vector[9] += 0.8L;
            vector[11] += 0.8L;
            vector[18] += 1.0L;
            vector[19] += 1.0L;
            vector[23] += 0.5L;
        }
        if ((has("sliding") || has("slid")) && has("window")) {
            vector[0] += 0.5L;
            vector[3] += 0.7L;
            vector[20] += 1.0L;
            vector[21] += 1.0L;
            vector[22] += 1.0L;
        }
        if (has("disjoint") || (has("union") && has("find"))) {
            vector[5] += 1.0L;
            vector[6] += 0.8L;
            vector[9] += 0.5L;
            vector[11] += 0.5L;
        }
        vector[14] += has("return") ? 0.5L : 0.2L;
        normalize_vector(vector);
        return vector;
    }

    static long double vector_similarity(const std::vector<long double>& left,
                                         const std::vector<long double>& right) {
        const std::size_t n = std::min(left.size(), right.size());
        long double dot = 0.0L;
        for (std::size_t i = 0; i < n; ++i) {
            dot += left[i] * right[i];
        }
        return std::clamp(dot, 0.0L, 1.0L);
    }

    static long double structural_body_pressure(const std::vector<std::string>& candidate_tokens,
                                                const std::vector<std::string>& salient) {
        std::size_t query_structural = 0;
        std::size_t matched = 0;
        for (const auto& token : salient) {
            if (!structural_signature_token(token)) {
                continue;
            }
            ++query_structural;
            if (std::find(candidate_tokens.begin(), candidate_tokens.end(), token) != candidate_tokens.end()) {
                ++matched;
            }
        }
        if (query_structural == 0) {
            return 0.0L;
        }
        return static_cast<long double>(matched) / static_cast<long double>(query_structural);
    }

    static bool static_literal_dump(const CodeSnippet& snippet) {
        const auto lower = lower_copy(snippet.code);
        if (lower.size() < 1200U) {
            return false;
        }
        const bool has_control_flow =
            lower.find("\n    for ") != std::string::npos ||
            lower.find("\n    while ") != std::string::npos ||
            lower.find("\n    if ") != std::string::npos ||
            lower.find(" raise ") != std::string::npos;
        if (has_control_flow) {
            return false;
        }
        std::size_t literal_lines = 0;
        std::stringstream stream(snippet.code);
        std::string line;
        while (std::getline(stream, line)) {
            const auto trimmed = trim(line);
            if (trimmed.find("\":") != std::string::npos ||
                trimmed.find("':") != std::string::npos ||
                trimmed.find("\",") != std::string::npos ||
                trimmed.find("',") != std::string::npos) {
                ++literal_lines;
            }
        }
        const bool literal_assignment =
            lower.find("return {") != std::string::npos ||
            lower.find(" = {") != std::string::npos ||
            lower.find(" = [") != std::string::npos ||
            lower.find("choices") != std::string::npos ||
            lower.find("integration-name") != std::string::npos;
        return literal_assignment && literal_lines >= 16U;
    }

    static bool non_algorithmic_bulk_snippet(const CodeSnippet& snippet) {
        if (snippet.code.size() < 1800U) {
            return false;
        }
        const auto lower = lower_copy(snippet.code);
        const bool has_dynamic_structure =
            lower.find("\n    for ") != std::string::npos ||
            lower.find("\n    while ") != std::string::npos ||
            lower.find("+=") != std::string::npos ||
            lower.find(".append(") != std::string::npos ||
            lower.find(" heap") != std::string::npos ||
            lower.find("visited") != std::string::npos ||
            lower.find("indegree") != std::string::npos ||
            lower.find("raise ") != std::string::npos;
        if (has_dynamic_structure) {
            return false;
        }
        const bool config_noise =
            lower.find("\"type\"") != std::string::npos ||
            lower.find("\"required\"") != std::string::npos ||
            lower.find("\"choices\"") != std::string::npos ||
            lower.find("integration-name") != std::string::npos ||
            lower.find("selector") != std::string::npos ||
            lower.find("provider") != std::string::npos;
        return config_noise;
    }

    static bool public_function_query(const std::vector<std::string>& query_tokens) {
        bool asks_function = false;
        bool asks_method = false;
        for (const auto& token : query_tokens) {
            if (token == "function" || token == "def") {
                asks_function = true;
            }
            if (token == "method" || token == "class" || token == "self" ||
                token == "internal" || token == "private") {
                asks_method = true;
            }
        }
        return asks_function && !asks_method;
    }

    static bool is_private_or_internal(std::string_view function_name,
                                       const std::string& code) {
        if (!function_name.empty() && function_name.front() == '_') {
            return true;
        }
        const auto lower = lower_copy(code);
        return lower.find("internal") != std::string::npos ||
               lower.find("private") != std::string::npos;
    }

    static bool is_method_like(const std::string& code) {
        const auto def = code.find("def ");
        if (def == std::string::npos) {
            return false;
        }
        const auto open = code.find('(', def);
        if (open == std::string::npos) {
            return false;
        }
        auto first_arg_start = code.find_first_not_of(" \t\r\n", open + 1U);
        if (first_arg_start == std::string::npos) {
            return false;
        }
        auto first_arg_end = code.find_first_of(",):", first_arg_start);
        if (first_arg_end == std::string::npos) {
            return false;
        }
        const auto arg = code.substr(first_arg_start, first_arg_end - first_arg_start);
        return arg == "self" || arg == "cls";
    }

    static bool is_indented_definition(const std::string& code) {
        const auto first = code.find_first_not_of("\r\n");
        return first != std::string::npos && (code[first] == ' ' || code[first] == '\t');
    }

    static std::string extract_function_name(const std::string& code) {
        const auto def = code.find("def ");
        if (def == std::string::npos) {
            return {};
        }
        std::size_t pos = def + 4U;
        std::string out;
        while (pos < code.size()) {
            const unsigned char ch = static_cast<unsigned char>(code[pos]);
            if (std::isalnum(ch) == 0 && code[pos] != '_') {
                break;
            }
            out.push_back(static_cast<char>(ch));
            ++pos;
        }
        return out;
    }

    static std::vector<std::string> extract_parameters(const std::string& code) {
        std::vector<std::string> params;
        const auto def = code.find("def ");
        if (def == std::string::npos) {
            return params;
        }
        const auto open = code.find('(', def);
        const auto close = code.find(')', open == std::string::npos ? def : open);
        if (open == std::string::npos || close == std::string::npos || close <= open) {
            return params;
        }
        std::stringstream stream(code.substr(open + 1U, close - open - 1U));
        std::string item;
        while (std::getline(stream, item, ',')) {
            item = trim(std::move(item));
            const auto colon = item.find(':');
            if (colon != std::string::npos) {
                item.resize(colon);
            }
            const auto equals = item.find('=');
            if (equals != std::string::npos) {
                item.resize(equals);
            }
            item = trim(std::move(item));
            while (!item.empty() && (item.front() == '*' || item.front() == '/')) {
                item.erase(item.begin());
            }
            item = trim(std::move(item));
            if (!item.empty()) {
                params.push_back(std::move(item));
            }
        }
        return params;
    }

    static std::string extract_return_expression(const std::string& code) {
        std::stringstream stream(code);
        std::string line;
        while (std::getline(stream, line)) {
            auto trimmed = ltrim(line);
            if (trimmed.rfind("return ", 0) == 0) {
                return trim(trimmed.substr(7U));
            }
        }
        return {};
    }

    static std::vector<std::string> infer_operations_from_expression(const std::string& expression) {
        std::vector<std::string> operations;
        auto trimmed = trim(expression);
        const auto open = trimmed.find('(');
        if (open != std::string::npos && open > 0U) {
            auto callee = trimmed.substr(0, open);
            const auto dot = callee.find_last_of('.');
            if (dot != std::string::npos) {
                callee = callee.substr(dot + 1U);
            }
            callee = sanitize_identifier_token(std::move(callee));
            if (safe_return_operation(callee)) {
                operations.push_back(std::move(callee));
            }
        }
        if (trimmed.find("[::-1]") != std::string::npos) {
            operations.push_back("reverse");
        }
        std::vector<std::string> unique;
        for (auto operation : operations) {
            if (!operation.empty() && std::find(unique.begin(), unique.end(), operation) == unique.end()) {
                unique.push_back(std::move(operation));
            }
        }
        return unique;
    }

    static bool safe_return_operation(std::string_view operation) {
        return operation == "sum" || operation == "len" || operation == "sorted" ||
               operation == "list" || operation == "tuple" || operation == "set" ||
               operation == "str" || operation == "int" || operation == "float" ||
               operation == "bool" || operation == "abs" || operation == "max" ||
               operation == "min" || operation == "reversed";
    }

    static std::string return_parameter(const std::vector<std::string>& parameters,
                                        const std::string& expression) {
        for (const auto& parameter : parameters) {
            if (parameter == "self" || parameter == "cls") {
                continue;
            }
            if (identifier_present(expression, parameter)) {
                return parameter;
            }
        }
        return {};
    }

    static std::string canonical_operation_template(std::string_view operation,
                                                    const std::string& expression_template) {
        if (operation == "sum") {
            return "sum({arg})";
        }
        if (operation == "len") {
            return "len({arg})";
        }
        if (operation == "sorted") {
            return "sorted({arg})";
        }
        if (operation == "list") {
            return "list({arg})";
        }
        if (operation == "tuple") {
            return "tuple({arg})";
        }
        if (operation == "set") {
            return "set({arg})";
        }
        if (operation == "str") {
            return "str({arg})";
        }
        if (operation == "int") {
            return "int({arg})";
        }
        if (operation == "float") {
            return "float({arg})";
        }
        if (operation == "bool") {
            return "bool({arg})";
        }
        if (operation == "abs") {
            return "abs({arg})";
        }
        if (operation == "max") {
            return "max({arg})";
        }
        if (operation == "min") {
            return "min({arg})";
        }
        if (operation == "reversed") {
            return "list(reversed({arg}))";
        }
        if (operation == "reverse") {
            return expression_template.find("{arg}") == std::string::npos ? std::string{} : expression_template;
        }
        return {};
    }

    static std::vector<std::string> normalize_tokens(const std::vector<std::string>& tokens) {
        std::vector<std::string> out;
        for (auto token : tokens) {
            token = normalize_token(std::move(token));
            if (!token.empty() &&
                std::find(out.begin(), out.end(), token) == out.end()) {
                out.push_back(std::move(token));
            }
        }
        return out;
    }

    static std::vector<std::string> salient_query_tokens(const std::vector<std::string>& query_tokens) {
        std::vector<std::string> out;
        for (auto token : query_tokens) {
            token = normalize_token(std::move(token));
            if (token.size() < 3U || is_code_query_stopword(token)) {
                continue;
            }
            if (std::find(out.begin(), out.end(), token) == out.end()) {
                out.push_back(std::move(token));
            }
        }
        return out;
    }

    static std::size_t indentation_width(std::string_view line) {
        std::size_t width = 0;
        for (const char ch : line) {
            if (ch == ' ') {
                ++width;
            } else if (ch == '\t') {
                width += 4U;
            } else {
                break;
            }
        }
        return width;
    }

    static std::vector<std::string> target_function_tokens(const std::vector<std::string>& query_tokens) {
        std::vector<std::string> out;
        for (auto token : query_tokens) {
            if ((token.size() < 3U && token != "is") || token == "write" || token == "python" ||
                token == "function" || token == "code" || token == "that" ||
                token == "this" || token == "with" || token == "from" ||
                token == "takes" || token == "take" || token == "return" ||
                token == "returns") {
                continue;
            }
            token = sanitize_identifier_token(std::move(token));
            if (!token.empty()) {
                out.push_back(std::move(token));
            }
            if (out.size() >= 4U) {
                break;
            }
        }
        return out;
    }

    static std::string infer_target_parameter(const std::vector<std::string>& query_tokens) {
        static const std::vector<std::string> parameter_words{
            "values", "value", "items", "item", "numbers", "number",
            "sequence", "string", "text", "data", "list"};
        for (auto it = query_tokens.rbegin(); it != query_tokens.rend(); ++it) {
            auto token = sanitize_identifier_token(*it);
            if (std::find(parameter_words.begin(), parameter_words.end(), token) != parameter_words.end()) {
                if (token == "list") {
                    return "values";
                }
                if (token == "string") {
                    return "text";
                }
                return token;
            }
        }
        return "value";
    }

    static std::string normalize_token(std::string token) {
        if (token.size() > 4U && token.back() == 's') {
            token.pop_back();
        }
        if (token.size() > 5U && token.ends_with("ing")) {
            token.resize(token.size() - 3U);
        }
        if (token.size() > 4U && token.ends_with("ed")) {
            token.resize(token.size() - 2U);
        }
        return token;
    }

    static std::vector<std::string> learned_name_triggers(const std::vector<std::string>& name_tokens) {
        std::vector<std::string> triggers;
        for (auto token : name_tokens) {
            token = normalize_token(std::move(token));
            if (token.size() < 3U || token == "value" || token == "values" ||
                token == "item" || token == "items" || token == "number" ||
                token == "numbers" || token == "list" || token == "sequence" ||
                token == "data" || token == "get" || token == "set" ||
                token == "make" || token == "build") {
                continue;
            }
            if (std::find(triggers.begin(), triggers.end(), token) == triggers.end()) {
                triggers.push_back(std::move(token));
            }
        }
        return triggers;
    }

    static bool predicate_trigger_token(std::string_view token) {
        return token.size() >= 3U &&
               token != "count" && token != "counts" &&
               token != "any" && token != "all" && token != "has" &&
               token != "have" && token != "with" && token != "without" &&
               token != "is" && token != "are" && token != "true" &&
               token != "false" && token != "filter" && token != "find" &&
               token != "check" && token != "value" && token != "values" &&
               token != "item" && token != "items" && token != "number" &&
               token != "numbers" && token != "list" && token != "sequence" &&
               token != "data";
    }

    static std::string first_loop_variable(const std::string& code) {
        std::stringstream stream{code};
        std::string line;
        while (std::getline(stream, line)) {
            auto trimmed = trim(line);
            if (trimmed.rfind("for ", 0) != 0) {
                continue;
            }
            const auto in_pos = trimmed.find(" in ");
            if (in_pos == std::string::npos || in_pos <= 4U) {
                continue;
            }
            auto variable = trim(trimmed.substr(4U, in_pos - 4U));
            if (variable.find(',') != std::string::npos || variable.find('(') != std::string::npos) {
                continue;
            }
            variable = sanitize_identifier_token(std::move(variable));
            if (!variable.empty()) {
                return variable;
            }
        }
        return {};
    }

    static std::string first_if_condition(const std::string& code) {
        std::stringstream stream{code};
        std::string line;
        while (std::getline(stream, line)) {
            auto trimmed = trim(line);
            if (trimmed.rfind("if ", 0) != 0) {
                continue;
            }
            auto condition = trim(trimmed.substr(3U));
            if (!condition.empty() && condition.back() == ':') {
                condition.pop_back();
            }
            condition = trim(std::move(condition));
            if (!condition.empty()) {
                return condition;
            }
        }
        return {};
    }

    static std::string first_append_argument(const std::string& code) {
        std::stringstream stream{code};
        std::string line;
        while (std::getline(stream, line)) {
            auto trimmed = trim(line);
            const auto append_pos = trimmed.find(".append(");
            if (append_pos == std::string::npos) {
                continue;
            }
            const auto start = append_pos + std::string_view(".append(").size();
            std::size_t depth = 1;
            for (std::size_t i = start; i < trimmed.size(); ++i) {
                if (trimmed[i] == '(') {
                    ++depth;
                } else if (trimmed[i] == ')') {
                    --depth;
                    if (depth == 0) {
                        return trim(trimmed.substr(start, i - start));
                    }
                }
            }
        }
        return {};
    }

    static std::pair<std::string, std::string> first_accumulator_update(const std::string& code,
                                                                        const std::string& loop_variable) {
        if (loop_variable.empty()) {
            return {};
        }
        std::stringstream stream{code};
        std::string line;
        while (std::getline(stream, line)) {
            auto trimmed = trim(line);
            const auto plus_equal = trimmed.find("+=");
            if (plus_equal == std::string::npos) {
                continue;
            }
            auto accumulator = sanitize_identifier_token(trim(trimmed.substr(0, plus_equal)));
            auto rhs = trim(trimmed.substr(plus_equal + 2U));
            if (accumulator.empty() || !identifier_present(rhs, loop_variable)) {
                continue;
            }
            rhs = replace_identifier(rhs, loop_variable, "{item}");
            return {accumulator, "{acc} += " + rhs};
        }
        return {};
    }

    static std::string first_assignment_value(const std::string& code,
                                              const std::string& variable) {
        if (variable.empty()) {
            return {};
        }
        std::stringstream stream{code};
        std::string line;
        while (std::getline(stream, line)) {
            auto trimmed = trim(line);
            if (trimmed.rfind("for ", 0) == 0) {
                break;
            }
            const auto equal = trimmed.find('=');
            if (equal == std::string::npos || trimmed.find("==") != std::string::npos) {
                continue;
            }
            auto lhs = sanitize_identifier_token(trim(trimmed.substr(0, equal)));
            if (lhs != variable) {
                continue;
            }
            auto rhs = trim(trimmed.substr(equal + 1U));
            if (!rhs.empty()) {
                return rhs;
            }
        }
        return {};
    }

    static std::string sanitize_identifier_token(std::string token) {
        std::string out;
        for (const unsigned char ch : token) {
            if (std::isalnum(ch) != 0 || ch == '_') {
                out.push_back(static_cast<char>(std::tolower(ch)));
            }
        }
        if (!out.empty() && std::isdigit(static_cast<unsigned char>(out.front())) != 0) {
            out.insert(out.begin(), 'v');
        }
        return out;
    }

    static std::string join_identifier(const std::vector<std::string>& tokens) {
        std::string out;
        for (const auto& token : tokens) {
            if (token.empty()) {
                continue;
            }
            if (!out.empty()) {
                out.push_back('_');
            }
            out += token;
        }
        return out.empty() ? std::string("generated") : out;
    }

    static std::string replace_identifier(std::string text,
                                          const std::string& from,
                                          const std::string& to) {
        if (from.empty() || from == to) {
            return text;
        }
        std::string out;
        out.reserve(text.size());
        for (std::size_t i = 0; i < text.size();) {
            const bool match = text.compare(i, from.size(), from) == 0;
            const bool left_ok = i == 0 || (std::isalnum(static_cast<unsigned char>(text[i - 1U])) == 0 && text[i - 1U] != '_');
            const bool right_ok = i + from.size() >= text.size() ||
                                  (std::isalnum(static_cast<unsigned char>(text[i + from.size()])) == 0 &&
                                   text[i + from.size()] != '_');
            if (match && left_ok && right_ok) {
                out += to;
                i += from.size();
            } else {
                out.push_back(text[i]);
                ++i;
            }
        }
        return out;
    }

    static bool identifier_present(const std::string& text,
                                   const std::string& identifier) {
        if (identifier.empty()) {
            return false;
        }
        for (std::size_t i = 0; i < text.size(); ++i) {
            const bool match = text.compare(i, identifier.size(), identifier) == 0;
            const bool left_ok = i == 0 || (std::isalnum(static_cast<unsigned char>(text[i - 1U])) == 0 && text[i - 1U] != '_');
            const bool right_ok = i + identifier.size() >= text.size() ||
                                  (std::isalnum(static_cast<unsigned char>(text[i + identifier.size()])) == 0 &&
                                   text[i + identifier.size()] != '_');
            if (match && left_ok && right_ok) {
                return true;
            }
        }
        return false;
    }

    static bool return_expression_is_self_contained(const std::string& expression,
                                                    const std::string& parameter) {
        bool saw_parameter = false;
        for (std::size_t i = 0; i < expression.size();) {
            const unsigned char ch = static_cast<unsigned char>(expression[i]);
            if (ch == '\'' || ch == '"') {
                const char quote = expression[i++];
                while (i < expression.size()) {
                    if (expression[i] == '\\') {
                        i += 2U;
                        continue;
                    }
                    if (expression[i++] == quote) {
                        break;
                    }
                }
                continue;
            }
            if (std::isalpha(ch) == 0 && ch != '_') {
                ++i;
                continue;
            }
            const std::size_t start = i;
            while (i < expression.size()) {
                const unsigned char token_ch = static_cast<unsigned char>(expression[i]);
                if (std::isalnum(token_ch) == 0 && token_ch != '_') {
                    break;
                }
                ++i;
            }
            const auto token = expression.substr(start, i - start);
            if (token == parameter) {
                saw_parameter = true;
                continue;
            }
            if (safe_expression_identifier(token)) {
                continue;
            }
            if (start > 0U && expression[start - 1U] == '.' &&
                dotted_expression_root(expression, start) == parameter) {
                continue;
            }
            return false;
        }
        return saw_parameter;
    }

    static bool template_expression_is_self_contained(std::string expression,
                                                      std::string_view placeholder,
                                                      std::string_view parameter) {
        bool replaced = false;
        while (true) {
            const auto marker = expression.find(placeholder);
            if (marker == std::string::npos) {
                break;
            }
            expression.replace(marker, placeholder.size(), parameter);
            replaced = true;
        }
        return replaced && return_expression_is_self_contained(expression, std::string(parameter));
    }

    static bool template_statement_is_self_contained(std::string statement) {
        bool saw_accumulator = false;
        bool saw_item = false;
        while (true) {
            const auto marker = statement.find("{acc}");
            if (marker == std::string::npos) {
                break;
            }
            statement.replace(marker, 5U, "total");
            saw_accumulator = true;
        }
        while (true) {
            const auto marker = statement.find("{item}");
            if (marker == std::string::npos) {
                break;
            }
            statement.replace(marker, 6U, "value");
            saw_item = true;
        }
        if (!saw_accumulator || !saw_item) {
            return false;
        }
        for (std::size_t i = 0; i < statement.size();) {
            const unsigned char ch = static_cast<unsigned char>(statement[i]);
            if (ch == '\'' || ch == '"') {
                const char quote = statement[i++];
                while (i < statement.size()) {
                    if (statement[i] == '\\') {
                        i += 2U;
                        continue;
                    }
                    if (statement[i++] == quote) {
                        break;
                    }
                }
                continue;
            }
            if (std::isalpha(ch) == 0 && ch != '_') {
                ++i;
                continue;
            }
            const std::size_t start = i;
            while (i < statement.size()) {
                const unsigned char token_ch = static_cast<unsigned char>(statement[i]);
                if (std::isalnum(token_ch) == 0 && token_ch != '_') {
                    break;
                }
                ++i;
            }
            const auto token = statement.substr(start, i - start);
            if (token == "total" || token == "value" || safe_expression_identifier(token)) {
                continue;
            }
            if (start > 0U && statement[start - 1U] == '.') {
                const auto root = dotted_expression_root(statement, start);
                if (root == "total" || root == "value") {
                    continue;
                }
            }
            return false;
        }
        return true;
    }

    static std::string dotted_expression_root(const std::string& expression,
                                              std::size_t dotted_member_start) {
        if (dotted_member_start < 2U || expression[dotted_member_start - 1U] != '.') {
            return {};
        }
        std::size_t pos = dotted_member_start - 1U;
        while (pos > 0U) {
            const unsigned char previous = static_cast<unsigned char>(expression[pos - 1U]);
            if (std::isalnum(previous) == 0 && previous != '_' && previous != '.') {
                break;
            }
            --pos;
        }
        while (pos < expression.size() && expression[pos] == '.') {
            ++pos;
        }
        const std::size_t root_start = pos;
        while (pos < expression.size()) {
            const unsigned char ch = static_cast<unsigned char>(expression[pos]);
            if (std::isalnum(ch) == 0 && ch != '_') {
                break;
            }
            ++pos;
        }
        return expression.substr(root_start, pos - root_start);
    }

    static bool safe_expression_identifier(std::string_view token) {
        return token == "None" || token == "True" || token == "False" ||
               token == "and" || token == "or" || token == "not" ||
               token == "is" || token == "in" ||
               safe_return_operation(token);
    }

    static bool is_code_query_stopword(std::string_view token) {
        return token == "write" || token == "python" || token == "function" ||
               token == "code" || token == "that" || token == "this" ||
               token == "with" || token == "from" || token == "take" ||
               token == "takes" || token == "return" || token == "returns" ||
               token == "input" || token == "output" || token == "value" ||
               token == "values" || token == "list" || token == "string" ||
               token == "text" || token == "number" || token == "integer";
    }

    static std::size_t shared_count(const std::vector<std::string>& left,
                                    const std::vector<std::string>& right) {
        std::size_t shared = 0;
        for (const auto& token : right) {
            if (std::find(left.begin(), left.end(), token) != left.end()) {
                ++shared;
            }
        }
        return shared;
    }

    static bool has_token(const std::vector<std::string>& tokens,
                          std::string_view token) {
        return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
    }

    static std::string ltrim(std::string_view text) {
        const auto first = text.find_first_not_of(" \t\r\n");
        return first == std::string_view::npos ? std::string{} : std::string(text.substr(first));
    }

    static std::string rtrim(std::string text) {
        const auto last = text.find_last_not_of(" \t\r\n");
        if (last == std::string::npos) {
            return {};
        }
        text.resize(last + 1U);
        return text;
    }

    static std::string trim(std::string text) {
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return {};
        }
        const auto last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1U);
    }

    static std::string lower_copy(std::string_view text) {
        std::string out(text);
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return out;
    }

    std::vector<CodeSnippet> snippets_;
    mutable std::unordered_map<std::string, std::vector<std::size_t>> token_index_;
    mutable bool token_index_dirty_ = false;
    std::unordered_map<std::string, long double> organ_strengths_;
    mutable std::vector<CodeOperationLink> operation_links_;
    mutable std::vector<CodePredicateMotif> predicate_motifs_;
    mutable std::vector<CodeTransformMotif> transform_motifs_;
    mutable std::vector<CodeReduceMotif> reduce_motifs_;
    mutable long double count_if_trajectory_strength_ = 0.0L;
    mutable long double collect_if_trajectory_strength_ = 0.0L;
    mutable long double map_transform_trajectory_strength_ = 0.0L;
    mutable long double reduce_accumulator_trajectory_strength_ = 0.0L;
    mutable bool operation_links_dirty_ = false;
    std::size_t max_snippets_;
};

} // namespace dzeta
