#pragma once

#include "dzeta_vm.h"
#include "field_state.h"
#include "information.h"
#include "working_memory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct EpisodeRecord {
    std::size_t id = 0;
    std::string timestamp;
    std::string prompt;
    std::string answer;
    long double confidence = 0.0L;
    bool success = false;
    std::string kind;
    std::uint64_t field_digest = 0;
};

struct EpisodeHit {
    EpisodeRecord episode;
    long double score = 0.0L;
};

inline std::size_t token_overlap_count(std::string_view left, std::string_view right) {
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

class EpisodicMemory {
public:
    std::size_t record(std::string timestamp,
                       std::string prompt,
                       std::string answer,
                       long double confidence,
                       bool success,
                       std::string kind) {
        EpisodeRecord episode;
        episode.id = next_id_++;
        episode.timestamp = std::move(timestamp);
        episode.prompt = std::move(prompt);
        episode.answer = std::move(answer);
        episode.confidence = std::clamp(confidence, 0.0L, 1.0L);
        episode.success = success;
        episode.kind = std::move(kind);
        episode.field_digest = stable_hash(episode.timestamp) ^ (stable_hash(episode.prompt) << 1U);
        episodes_.push_back(std::move(episode));
        return episodes_.back().id;
    }

    std::optional<EpisodeRecord> retrieve_by_time(std::string_view timestamp) const {
        for (const auto& episode : episodes_) {
            if (episode.timestamp == timestamp) {
                return episode;
            }
        }
        return std::nullopt;
    }

    std::vector<EpisodeHit> retrieve_by_relevance(std::string_view context, std::size_t limit) const {
        std::vector<EpisodeHit> hits;
        const auto context_tokens = tokenize_query(context);
        for (const auto& episode : episodes_) {
            const auto overlap = static_cast<long double>(token_overlap_count(context, episode.prompt + " " + episode.answer));
            const long double lexical = overlap / static_cast<long double>(std::max<std::size_t>(1, context_tokens.size()));
            const long double recency = episodes_.empty()
                                            ? 0.0L
                                            : static_cast<long double>(episode.id + 1U) / static_cast<long double>(next_id_);
            EpisodeHit hit;
            hit.episode = episode;
            hit.score = std::clamp(0.62L * lexical + 0.26L * episode.confidence + 0.12L * recency,
                                   0.0L, 1.0L);
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

    const std::vector<EpisodeRecord>& episodes() const noexcept {
        return episodes_;
    }

private:
    std::size_t next_id_ = 1;
    std::vector<EpisodeRecord> episodes_;

    friend void save_episodic_memory(const EpisodicMemory&, std::string_view);
    friend EpisodicMemory load_episodic_memory(std::string_view);
};

inline std::string agi_hex_encode(std::string_view text) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(text.size() * 2U);
    for (unsigned char ch : text) {
        out.push_back(hex[(ch >> 4U) & 0x0FU]);
        out.push_back(hex[ch & 0x0FU]);
    }
    return out;
}

inline int agi_hex_value(char ch) {
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

inline std::string agi_hex_decode(std::string_view text) {
    std::string out;
    out.reserve(text.size() / 2U);
    for (std::size_t i = 0; i + 1U < text.size(); i += 2U) {
        const int hi = agi_hex_value(text[i]);
        const int lo = agi_hex_value(text[i + 1U]);
        if (hi < 0 || lo < 0) {
            return {};
        }
        out.push_back(static_cast<char>((hi << 4U) | lo));
    }
    return out;
}

inline void save_episodic_memory(const EpisodicMemory& memory, std::string_view path) {
    std::ofstream output(std::string(path), std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write episodic memory: " + std::string(path));
    }
    output << "DZETA_EPISODIC_MEMORY_V1\n";
    for (const auto& episode : memory.episodes_) {
        output << episode.id << '\t'
               << agi_hex_encode(episode.timestamp) << '\t'
               << agi_hex_encode(episode.prompt) << '\t'
               << agi_hex_encode(episode.answer) << '\t'
               << static_cast<double>(episode.confidence) << '\t'
               << (episode.success ? 1 : 0) << '\t'
               << agi_hex_encode(episode.kind) << '\t'
               << episode.field_digest << '\n';
    }
}

inline EpisodicMemory load_episodic_memory(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read episodic memory: " + std::string(path));
    }
    std::string header;
    std::getline(input, header);
    if (header != "DZETA_EPISODIC_MEMORY_V1") {
        throw std::runtime_error("invalid episodic memory: " + std::string(path));
    }
    EpisodicMemory memory;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, '\t')) {
            fields.push_back(field);
        }
        if (fields.size() != 8) {
            throw std::runtime_error("invalid episodic memory row: " + std::string(path));
        }
        EpisodeRecord episode;
        episode.id = static_cast<std::size_t>(std::stoull(fields[0]));
        episode.timestamp = agi_hex_decode(fields[1]);
        episode.prompt = agi_hex_decode(fields[2]);
        episode.answer = agi_hex_decode(fields[3]);
        episode.confidence = std::stold(fields[4]);
        episode.success = fields[5] == "1";
        episode.kind = agi_hex_decode(fields[6]);
        episode.field_digest = static_cast<std::uint64_t>(std::stoull(fields[7]));
        memory.next_id_ = std::max(memory.next_id_, episode.id + 1U);
        memory.episodes_.push_back(std::move(episode));
    }
    return memory;
}

struct AttentionEpisode {
    EpisodeRecord episode;
    long double score = 0.0L;
};

class AttentionField {
public:
    std::vector<AttentionEpisode> attend_episodes(std::string_view query,
                                                  const std::vector<EpisodeRecord>& episodes,
                                                  std::size_t top_k) const {
        std::vector<AttentionEpisode> out;
        const auto query_tokens = tokenize_query(query);
        for (const auto& episode : episodes) {
            const long double overlap = static_cast<long double>(token_overlap_count(query, episode.prompt + " " + episode.answer));
            AttentionEpisode item;
            item.episode = episode;
            item.score = std::clamp(0.70L * overlap / static_cast<long double>(std::max<std::size_t>(1, query_tokens.size())) +
                                    0.30L * episode.confidence,
                                    0.0L, 1.0L);
            out.push_back(std::move(item));
        }
        std::sort(out.begin(), out.end(), [](const auto& left, const auto& right) {
            return left.score > right.score;
        });
        if (out.size() > top_k) {
            out.resize(top_k);
        }
        return out;
    }
};

struct MetaCognitionReport {
    long double confidence = 0.0L;
    std::vector<std::string> known_gaps;
    bool should_defer = false;
};

inline MetaCognitionReport evaluate_metacognition(const std::vector<std::string>& candidates,
                                                  const std::vector<std::string>& evidence,
                                                  long double uncertainty,
                                                  long double defer_threshold) {
    MetaCognitionReport report;
    const long double candidate_support = std::min<long double>(1.0L, static_cast<long double>(candidates.size()) / 3.0L);
    const long double evidence_support = std::min<long double>(1.0L, static_cast<long double>(evidence.size()) / 2.0L);
    report.confidence = std::clamp(0.42L * candidate_support + 0.48L * evidence_support + 0.10L * (1.0L - uncertainty),
                                   0.0L, 1.0L);
    if (candidates.empty()) {
        report.known_gaps.push_back("no candidate solution");
    }
    if (evidence.empty()) {
        report.known_gaps.push_back("no supporting memory or verification");
    }
    if (uncertainty > 0.65L) {
        report.known_gaps.push_back("high uncertainty");
    }
    report.should_defer = report.confidence < defer_threshold;
    return report;
}

class TheoryOfMind {
public:
    void observe_user_belief(std::string subject, std::string value, long double confidence) {
        user_beliefs_.push_back({std::move(subject), std::move(value), std::clamp(confidence, 0.0L, 1.0L)});
    }

    void observe_reality(std::string subject, std::string value, long double confidence) {
        reality_.push_back({std::move(subject), std::move(value), std::clamp(confidence, 0.0L, 1.0L)});
    }

    bool has_false_belief(std::string_view subject) const {
        for (const auto& belief : user_beliefs_) {
            if (belief.subject != subject || belief.confidence < 0.5L) {
                continue;
            }
            for (const auto& real : reality_) {
                if (real.subject == subject && real.confidence > 0.5L && real.value != belief.value) {
                    return true;
                }
            }
        }
        return false;
    }

private:
    struct Belief {
        std::string subject;
        std::string value;
        long double confidence = 0.0L;
    };
    std::vector<Belief> user_beliefs_;
    std::vector<Belief> reality_;
};

struct CounterfactualLink {
    std::string cause;
    std::string effect;
    long double strength = 0.0L;
};

struct CounterfactualResult {
    std::string cause;
    std::string effect;
    long double baseline_probability = 0.0L;
    long double outcome_probability = 0.0L;
    long double delta = 0.0L;
};

class CounterfactualModel {
public:
    void observe(std::string cause, std::string effect, long double strength) {
        for (auto& link : links_) {
            if (link.cause == cause && link.effect == effect) {
                link.strength = std::clamp(link.strength + 0.15L * strength, 0.0L, 1.0L);
                return;
            }
        }
        links_.push_back({std::move(cause), std::move(effect), std::clamp(strength, 0.0L, 1.0L)});
    }

    CounterfactualResult simulate(std::string_view effect, std::string_view cause, bool do_activate) const {
        CounterfactualResult result;
        result.cause = std::string(cause);
        result.effect = std::string(effect);
        result.baseline_probability = 0.10L;
        result.outcome_probability = result.baseline_probability;
        for (const auto& link : links_) {
            if (link.effect != effect) {
                continue;
            }
            result.baseline_probability = std::max(result.baseline_probability, 0.20L + 0.50L * link.strength);
            if (link.cause == cause) {
                result.outcome_probability = do_activate
                                                 ? std::max(result.outcome_probability, 0.25L + 0.70L * link.strength)
                                                 : std::min(result.outcome_probability, 0.12L + 0.10L * (1.0L - link.strength));
            }
        }
        result.delta = std::abs(result.outcome_probability - result.baseline_probability);
        return result;
    }

private:
    std::vector<CounterfactualLink> links_;
};

struct CausalTableRow {
    int parent_a = 0;
    int parent_b = 0;
    long double probability = 0.0L;
};

struct StructuralCausalNode {
    std::string name;
    std::vector<std::string> parents;
    long double root_probability = 0.0L;
    std::vector<CausalTableRow> table;
    bool intervened = false;
    bool intervention_value = false;
};

struct StructuralCounterfactualReport {
    long double factual_probability = 0.0L;
    long double counterfactual_probability = 0.0L;
};

class StructuralCausalModel {
public:
    void add_binary_root(std::string name, long double probability) {
        StructuralCausalNode node;
        node.name = std::move(name);
        node.root_probability = std::clamp(probability, 0.0L, 1.0L);
        nodes_[node.name] = std::move(node);
    }

    void add_binary_child(std::string name,
                          std::vector<std::string> parents,
                          std::vector<CausalTableRow> table) {
        StructuralCausalNode node;
        node.name = std::move(name);
        node.parents = std::move(parents);
        node.table = std::move(table);
        nodes_[node.name] = std::move(node);
    }

    long double probability(std::string_view variable) const {
        return probability_with_assignments(variable, {});
    }

    StructuralCausalModel do_intervention(std::string_view variable, bool value) const {
        auto copy = *this;
        auto found = copy.nodes_.find(std::string(variable));
        if (found != copy.nodes_.end()) {
            found->second.intervened = true;
            found->second.intervention_value = value;
        }
        return copy;
    }

    StructuralCounterfactualReport counterfactual(std::string_view outcome,
                                                  std::string_view intervention,
                                                  bool value,
                                                  const std::vector<std::pair<std::string, bool>>& evidence) const {
        StructuralCounterfactualReport report;
        std::map<std::string, bool> assignments;
        for (const auto& [name, observed] : evidence) {
            if (name == outcome) {
                continue;
            }
            assignments[name] = observed;
        }
        report.factual_probability = probability_with_assignments(outcome, assignments);
        report.counterfactual_probability = do_intervention(intervention, value)
                                                 .probability_with_assignments(outcome, assignments);
        return report;
    }

private:
    long double probability_with_assignments(std::string_view variable,
                                             const std::map<std::string, bool>& assignments) const {
        const auto found = nodes_.find(std::string(variable));
        if (found == nodes_.end()) {
            return 0.0L;
        }
        const auto& node = found->second;
        if (node.intervened) {
            return node.intervention_value ? 1.0L : 0.0L;
        }
        const auto assigned = assignments.find(node.name);
        if (assigned != assignments.end()) {
            return assigned->second ? 1.0L : 0.0L;
        }
        if (node.parents.empty()) {
            return node.root_probability;
        }

        long double total = 0.0L;
        for (const auto& row : node.table) {
            long double row_probability = row.probability;
            for (std::size_t i = 0; i < node.parents.size(); ++i) {
                const bool parent_value = i == 0 ? row.parent_a != 0 : row.parent_b != 0;
                const long double parent_probability = probability_with_assignments(node.parents[i], assignments);
                row_probability *= parent_value ? parent_probability : (1.0L - parent_probability);
            }
            total += row_probability;
        }
        return std::clamp(total, 0.0L, 1.0L);
    }

    std::map<std::string, StructuralCausalNode> nodes_;
};

struct BayesianConcept {
    std::string name;
    std::string parent;
    std::map<std::string, long double> property_alpha;
    std::map<std::string, long double> property_beta;
};

class BayesianWorldModel {
public:
    void add_concept(std::string name, std::string parent = {}) {
        auto& node = concepts_[name];
        node.name = std::move(name);
        node.parent = std::move(parent);
    }

    void observe_property(std::string_view concept_name,
                          std::string_view property,
                          bool value,
                          long double confidence) {
        auto& node = concepts_[std::string(concept_name)];
        node.name = std::string(concept_name);
        const long double evidence = std::clamp(confidence, 0.0L, 1.0L);
        node.property_alpha[std::string(property)] += value ? evidence : 0.0L;
        node.property_beta[std::string(property)] += value ? 0.0L : evidence;
    }

    long double posterior(std::string_view concept_name,
                          std::string_view property,
                          std::string_view fallback_parent = {}) const {
        const auto found = concepts_.find(std::string(concept_name));
        if (found == concepts_.end()) {
            return fallback_parent.empty() ? 0.5L : posterior(fallback_parent, property);
        }
        const auto& node = found->second;
        const auto a_found = node.property_alpha.find(std::string(property));
        const auto b_found = node.property_beta.find(std::string(property));
        const long double alpha = 0.5L + (a_found == node.property_alpha.end() ? 0.0L : a_found->second);
        const long double beta = 0.5L + (b_found == node.property_beta.end() ? 0.0L : b_found->second);
        const bool has_local = alpha + beta > 1.0001L;
        const long double local = alpha / (alpha + beta);
        if (node.parent.empty()) {
            return local;
        }
        const long double inherited = posterior(node.parent, property);
        return has_local ? 0.78L * local + 0.22L * inherited : inherited;
    }

private:
    std::map<std::string, BayesianConcept> concepts_;
};

struct DecisionAction {
    std::string action;
    long double probability = 0.0L;
    long double utility = 0.0L;
    std::vector<DecisionAction> children;
};

struct DecisionSelection {
    std::string action;
    long double expected_utility = 0.0L;
};

class DecisionTreePlanner {
public:
    void add_action(std::string action, long double probability, long double utility) {
        actions_.push_back({std::move(action), std::clamp(probability, 0.0L, 1.0L), utility, {}});
    }

    DecisionSelection select_best() const {
        DecisionSelection best;
        best.expected_utility = -1.0e9L;
        for (const auto& action : actions_) {
            const long double expected = expected_utility(action);
            if (expected > best.expected_utility) {
                best.action = action.action;
                best.expected_utility = expected;
            }
        }
        return best;
    }

private:
    static long double expected_utility(const DecisionAction& action) {
        long double child_bonus = 0.0L;
        for (const auto& child : action.children) {
            child_bonus = std::max(child_bonus, expected_utility(child));
        }
        return action.probability * action.utility + 0.25L * child_bonus;
    }

    std::vector<DecisionAction> actions_;
};

struct AbstractionLevel {
    std::string name;
    std::vector<std::string> units;
};

struct AbstractionHierarchy {
    std::vector<AbstractionLevel> levels;

    std::optional<AbstractionLevel> find_level(std::string_view name) const {
        for (const auto& level : levels) {
            if (level.name == name) {
                return level;
            }
        }
        return std::nullopt;
    }
};

inline AbstractionHierarchy build_abstraction_hierarchy(std::string_view text) {
    AbstractionHierarchy hierarchy;
    const auto tokens = tokenize_query(text);
    hierarchy.levels.push_back({"token", tokens});

    std::vector<std::string> phrases;
    for (std::size_t i = 0; i < tokens.size(); i += 2U) {
        std::string phrase = tokens[i];
        if (i + 1U < tokens.size()) {
            phrase += " " + tokens[i + 1U];
        }
        phrases.push_back(std::move(phrase));
    }
    hierarchy.levels.push_back({"phrase", phrases});

    std::vector<std::string> concepts;
    for (const auto& phrase : phrases) {
        if (phrase.find("open") != std::string::npos || phrase.find("parse") != std::string::npos ||
            phrase.find("count") != std::string::npos || phrase.find("return") != std::string::npos) {
            concepts.push_back("data-processing");
        } else {
            concepts.push_back("context-cluster");
        }
    }
    std::sort(concepts.begin(), concepts.end());
    concepts.erase(std::unique(concepts.begin(), concepts.end()), concepts.end());
    hierarchy.levels.push_back({"concept", concepts});

    std::vector<std::string> procedures;
    if (tokens.size() >= 3) {
        procedures.push_back("ordered-transform");
    }
    if (std::find(tokens.begin(), tokens.end(), "return") != tokens.end()) {
        procedures.push_back("return-output");
    }
    hierarchy.levels.push_back({"procedure", procedures.empty() ? std::vector<std::string>{"single-step"} : procedures});

    std::vector<std::string> strategies;
    strategies.push_back(procedures.size() > 1 ? "read-transform-report" : "direct-response");
    hierarchy.levels.push_back({"strategy", strategies});
    return hierarchy;
}

struct EmotionState {
    long double valence = 0.0L;
    long double arousal = 0.35L;
    long double dominance = 0.50L;

    void update_from_outcome(long double reward, long double surprise) {
        reward = std::clamp(reward, -1.0L, 1.0L);
        surprise = std::clamp(surprise, 0.0L, 1.0L);
        valence = std::clamp(0.70L * valence + 0.30L * reward, -1.0L, 1.0L);
        arousal = std::clamp(0.65L * arousal + 0.35L * surprise, 0.0L, 1.0L);
        dominance = std::clamp(0.70L * dominance + 0.30L * (reward > 0.0L ? 0.75L : 0.25L), 0.0L, 1.0L);
    }

    long double exploration_temperature() const {
        return std::clamp(0.05L + 0.55L * arousal + 0.20L * std::max(0.0L, -valence), 0.01L, 1.0L);
    }
};

struct AgiFoundryConfig {
    std::size_t working_memory_slots = 7;
    std::size_t attention_top_k = 5;
    std::size_t vm_steps = 4096;
    long double defer_threshold = 0.68L;
    bool print_trace = false;
};

struct SelfTrainingTask {
    std::string prompt;
    long double novelty = 0.0L;
    long double compression_gain = 0.0L;
};

class IntrinsicMotivation {
public:
    std::vector<SelfTrainingTask> propose_self_training_tasks(const EpisodicMemory& memory,
                                                              const MetaCognitionReport& meta,
                                                              std::size_t limit) const {
        std::vector<SelfTrainingTask> tasks;
        for (const auto& gap : meta.known_gaps) {
            SelfTrainingTask task;
            task.prompt = "self-train gap: " + gap;
            task.novelty = 0.65L + 0.05L * static_cast<long double>(tasks.size());
            task.compression_gain = std::clamp(1.0L / static_cast<long double>(memory.episodes().size() + 1U),
                                               0.05L, 1.0L);
            tasks.push_back(std::move(task));
            if (tasks.size() >= limit) {
                return tasks;
            }
        }
        for (const auto& episode : memory.episodes()) {
            if (!episode.success) {
                tasks.push_back({"self-train failed episode " + std::to_string(episode.id),
                                 0.50L + (1.0L - episode.confidence) * 0.30L,
                                 0.20L});
                if (tasks.size() >= limit) {
                    break;
                }
            }
        }
        return tasks;
    }
};

struct RuntimePatch {
    bool changed = false;
    long double defer_threshold = 0.0L;
    std::size_t attention_top_k = 0;
    std::string reason;
};

class CognitiveArchitecture {
public:
    void enable(std::string component) {
        components_.insert(std::move(component));
    }

    bool enabled(std::string_view component) const {
        return components_.find(std::string(component)) != components_.end();
    }

    const std::set<std::string>& components() const noexcept {
        return components_;
    }

private:
    std::set<std::string> components_;
};

struct ArchitecturePatch {
    bool changed = false;
    std::vector<std::string> enable_components;
    std::string reason;
};

class ContinualLearningGuard {
public:
    void protect(std::string key, long double importance) {
        importance_[key] = std::clamp(importance, 0.0L, 1.0L);
        rehearsal_.push_back(std::move(key));
        sort_rehearsal();
    }

    void observe_new_trace(std::string key, long double importance) {
        importance_[key] = std::clamp(importance, 0.0L, 1.0L);
        if (importance > 0.25L) {
            rehearsal_.push_back(std::move(key));
            sort_rehearsal();
        }
    }

    bool should_retain(std::string_view key) const {
        const auto found = importance_.find(std::string(key));
        return found != importance_.end() && found->second >= 0.70L;
    }

    std::vector<std::string> rehearsal_items(std::size_t limit) const {
        auto out = rehearsal_;
        if (out.size() > limit) {
            out.resize(limit);
        }
        return out;
    }

private:
    void sort_rehearsal() {
        std::sort(rehearsal_.begin(), rehearsal_.end(), [&](const auto& left, const auto& right) {
            return importance_[left] > importance_[right];
        });
        rehearsal_.erase(std::unique(rehearsal_.begin(), rehearsal_.end()), rehearsal_.end());
    }

    std::map<std::string, long double> importance_;
    std::vector<std::string> rehearsal_;
};

struct MultimodalHit {
    std::string modality;
    std::string caption;
    long double score = 0.0L;
};

class MultimodalField {
public:
    void observe(std::string modality, std::vector<std::uint8_t> bytes, std::string caption) {
        records_.push_back({std::move(modality), std::move(bytes), std::move(caption)});
    }

    std::vector<MultimodalHit> retrieve(std::string_view query, std::size_t limit) const {
        std::vector<MultimodalHit> hits;
        const auto query_tokens = tokenize_query(query);
        for (const auto& record : records_) {
            const auto overlap = static_cast<long double>(token_overlap_count(query, record.caption));
            const long double byte_signal = std::min<long double>(1.0L, static_cast<long double>(record.bytes.size()) / 16.0L);
            MultimodalHit hit;
            hit.modality = record.modality;
            hit.caption = record.caption;
            hit.score = std::clamp(0.78L * overlap / static_cast<long double>(std::max<std::size_t>(1, query_tokens.size())) +
                                   0.22L * byte_signal,
                                   0.0L, 1.0L);
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

private:
    struct Record {
        std::string modality;
        std::vector<std::uint8_t> bytes;
        std::string caption;
    };
    std::vector<Record> records_;
};

struct InternetFetchResult {
    bool ok = false;
    std::string body;
    std::string error;
};

class InternetConnector {
public:
    void add_cached_document(std::string url, std::string body) {
        cache_[std::move(url)] = std::move(body);
    }

    void enable_live_fetch(bool enabled = true) noexcept {
        live_enabled_ = enabled;
    }

    InternetFetchResult fetch(std::string_view url) const {
        const auto found = cache_.find(std::string(url));
        if (found == cache_.end()) {
            if (!live_enabled_) {
                return {false, {}, "not in cache; live network adapter disabled"};
            }
            auto live = fetch_live(url);
            if (live.ok) {
                cache_[std::string(url)] = live.body;
            }
            return live;
        }
        return {true, found->second, {}};
    }

private:
    static bool allowed_url(std::string_view url) {
        return (url.starts_with("https://") || url.starts_with("http://")) && url.size() <= 2048U;
    }

    static std::string quote_for_powershell(std::string_view value) {
        std::string out;
        out.reserve(value.size() + 2U);
        out.push_back('\'');
        for (const char ch : value) {
            if (ch == '\'') {
                out += "''";
            } else {
                out.push_back(ch);
            }
        }
        out.push_back('\'');
        return out;
    }

    static std::string quote_for_sh(std::string_view value) {
        std::string out;
        out.reserve(value.size() + 2U);
        out.push_back('\'');
        for (const char ch : value) {
            if (ch == '\'') {
                out += "'\\''";
            } else {
                out.push_back(ch);
            }
        }
        out.push_back('\'');
        return out;
    }

    static InternetFetchResult run_fetch_command(const std::string& command) {
#ifdef _WIN32
        FILE* pipe = _popen(command.c_str(), "r");
#else
        FILE* pipe = popen(command.c_str(), "r");
#endif
        if (pipe == nullptr) {
            return {false, {}, "could not start network fetch process"};
        }
        std::array<char, 4096> buffer{};
        std::string body;
        constexpr std::size_t max_bytes = 512U * 1024U;
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            body.append(buffer.data());
            if (body.size() >= max_bytes) {
                body.resize(max_bytes);
                break;
            }
        }
#ifdef _WIN32
        const int code = _pclose(pipe);
#else
        const int code = pclose(pipe);
#endif
        if (code != 0 || body.empty()) {
            return {false, body, "network fetch failed"};
        }
        return {true, body, {}};
    }

    static InternetFetchResult fetch_live(std::string_view url) {
        if (!allowed_url(url)) {
            return {false, {}, "unsupported or oversized URL"};
        }
#ifdef _WIN32
        const std::string command =
            "powershell -NoProfile -ExecutionPolicy Bypass -Command "
            "\"$ProgressPreference='SilentlyContinue'; "
            "(Invoke-WebRequest -UseBasicParsing -TimeoutSec 10 -Uri " +
            quote_for_powershell(url) + ").Content\"";
#else
        const std::string command = "curl -L --max-time 10 --silent " + quote_for_sh(url);
#endif
        return run_fetch_command(command);
    }

    mutable std::map<std::string, std::string> cache_;
    bool live_enabled_ = false;
};

class SelfModificationEngine {
public:
    RuntimePatch propose_patch(const AgiFoundryConfig& config,
                               const MetaCognitionReport& meta,
                               const EmotionState& emotion) const {
        RuntimePatch patch;
        patch.defer_threshold = config.defer_threshold;
        patch.attention_top_k = config.attention_top_k;
        if (meta.should_defer && !meta.known_gaps.empty()) {
            patch.changed = true;
            patch.defer_threshold = std::max(0.35L, config.defer_threshold - 0.08L);
            patch.attention_top_k = std::min<std::size_t>(32, config.attention_top_k + 2U);
            patch.reason = "defer gap requires broader retrieval";
        }
        if (emotion.exploration_temperature() > 0.55L) {
            patch.changed = true;
            patch.attention_top_k = std::min<std::size_t>(32, patch.attention_top_k + 1U);
            if (patch.reason.empty()) {
                patch.reason = "high arousal increases exploration";
            }
        }
        return patch;
    }

    void apply(AgiFoundryConfig& config, const RuntimePatch& patch) const {
        if (!patch.changed) {
            return;
        }
        config.defer_threshold = patch.defer_threshold;
        config.attention_top_k = patch.attention_top_k;
    }

    ArchitecturePatch propose_architecture_patch(const CognitiveArchitecture& architecture,
                                                 const MetaCognitionReport& meta) const {
        ArchitecturePatch patch;
        if (meta.should_defer) {
            if (!architecture.enabled("counterfactual_probe")) {
                patch.enable_components.push_back("counterfactual_probe");
            }
            if (!architecture.enabled("episodic_rehearsal")) {
                patch.enable_components.push_back("episodic_rehearsal");
            }
        }
        patch.changed = !patch.enable_components.empty();
        if (patch.changed) {
            patch.reason = "knowledge gap requires new cognitive probes";
        }
        return patch;
    }

    void apply(CognitiveArchitecture& architecture, const ArchitecturePatch& patch) const {
        for (const auto& component : patch.enable_components) {
            architecture.enable(component);
        }
    }
};

struct AgiFoundryResult {
    std::string answer;
    bool deferred = false;
    long double confidence = 0.0L;
    std::string trace;
    std::size_t saved_episode_id = 0;
};

inline std::optional<std::string> extract_iso_timestamp(std::string_view text) {
    const std::string value(text);
    for (std::size_t i = 0; i + 19U <= value.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(value[i])) != 0 &&
            i + 19U <= value.size() &&
            value[i + 4U] == '-' && value[i + 7U] == '-' &&
            (value[i + 10U] == 'T' || value[i + 10U] == ' ')) {
            std::string timestamp = value.substr(i, 19U);
            if (i + 20U <= value.size() && value[i + 19U] == 'Z') {
                timestamp.push_back('Z');
            }
            return timestamp;
        }
    }
    return std::nullopt;
}

inline std::vector<std::pair<std::string, std::string>> parse_arrow_examples(std::string_view prompt) {
    std::vector<std::pair<std::string, std::string>> examples;
    std::string text(prompt);
    std::replace(text.begin(), text.end(), ';', '\n');
    std::stringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        const auto arrow = line.find("->");
        if (arrow == std::string::npos) {
            continue;
        }
        auto left = line.substr(0, arrow);
        auto right = line.substr(arrow + 2U);
        const auto colon = left.rfind(':');
        if (colon != std::string::npos) {
            left = left.substr(colon + 1U);
        }
        auto trim = [](std::string value) {
            const auto begin = value.find_first_not_of(" \t\r\n");
            const auto end = value.find_last_not_of(" \t\r\n");
            if (begin == std::string::npos) {
                return std::string{};
            }
            return value.substr(begin, end - begin + 1U);
        };
        left = trim(left);
        right = trim(right);
        if (!left.empty() && !right.empty()) {
            examples.push_back({left, right});
        }
    }
    return examples;
}

inline std::optional<std::string> parse_task_input(std::string_view prompt) {
    const std::string text(prompt);
    const auto task = text.find("task:");
    if (task == std::string::npos) {
        return std::nullopt;
    }
    std::string value = text.substr(task + 5U);
    const auto begin = value.find_first_not_of(" \t\r\n");
    const auto end = value.find_last_not_of(" \t\r\n.;");
    if (begin == std::string::npos) {
        return std::nullopt;
    }
    return value.substr(begin, end - begin + 1U);
}

class AgiFoundry {
public:
    explicit AgiFoundry(AgiFoundryConfig config = {})
        : config_(config), working_(config.working_memory_slots) {}

    AgiFoundry(AgiFoundryConfig config, EpisodicMemory memory, SkillGenome skills = {})
        : config_(config),
          memory_(std::move(memory)),
          working_(config.working_memory_slots),
          skills_(std::move(skills)) {}

    void observe_user_event(std::string timestamp, std::string utterance) {
        memory_.record(std::move(timestamp), std::move(utterance), "", 0.92L, true, "user_event");
    }

    AgiFoundryResult solve(std::string_view prompt) {
        AgiFoundryResult result;
        std::ostringstream trace;
        working_.remember("goal:" + std::string(prompt), 0.80L);

        const auto timestamp = extract_iso_timestamp(prompt);
        if (timestamp) {
            const auto episode = memory_.retrieve_by_time(*timestamp);
            if (episode) {
                result.answer = episode->prompt;
                result.confidence = std::max(0.80L, episode->confidence);
                result.saved_episode_id = memory_.record(*timestamp, std::string(prompt), result.answer, result.confidence, true, "recall");
                trace << "episode_id=" << episode->id << "\n";
                trace << "working_memory=" << working_.slots().size() << "\n";
                result.trace = trace.str();
                return result;
            }
        }

        const auto examples = parse_arrow_examples(prompt);
        const auto task_input = parse_task_input(prompt);
        if (task_input) {
            const auto lower_prompt = [prompt]() {
                std::string out(prompt);
                std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
                    return static_cast<char>(std::tolower(ch));
                });
                return out;
            }();
            for (const auto& record : skills_.records()) {
                if (lower_prompt.find(record.name) == std::string::npos &&
                    lower_prompt.find(record.program.name) == std::string::npos) {
                    continue;
                }
                const auto run = vm_.execute(record.program, *task_input, config_.vm_steps);
                if (run.ok) {
                    result.answer = run.output;
                    result.confidence = std::max(0.80L, record.confidence);
                    result.saved_episode_id = memory_.record("now", std::string(prompt), result.answer, result.confidence, true, "stored_skill");
                    trace << "stored_skill=" << record.name << "\n";
                    trace << "vm_verified=yes\n";
                    trace << "working_memory=" << working_.slots().size() << "\n";
                    result.trace = trace.str();
                    return result;
                }
            }
        }
        if (!examples.empty() && task_input) {
            const auto program = synthesizer_.synthesize(examples);
            if (program) {
                const auto run = vm_.execute(*program, *task_input, config_.vm_steps);
                if (run.ok) {
                    skills_.store(program->name, *program, 0.90L);
                    result.answer = run.output;
                    result.confidence = 0.90L;
                    result.saved_episode_id = memory_.record("now", std::string(prompt), result.answer, result.confidence, true, "vm_skill");
                    trace << "candidate_program=" << program->name << "\n";
                    trace << "vm_verified=yes\n";
                    trace << "working_memory=" << working_.slots().size() << "\n";
                    result.trace = trace.str();
                    return result;
                }
            }
        }

        const auto attended = attention_.attend_episodes(prompt, memory_.episodes(), config_.attention_top_k);
        std::vector<std::string> candidates;
        std::vector<std::string> evidence;
        if (!attended.empty() && attended.front().score > 0.62L) {
            candidates.push_back(attended.front().episode.answer.empty()
                                     ? attended.front().episode.prompt
                                     : attended.front().episode.answer);
            evidence.push_back("episodic memory");
        }
        const auto meta = evaluate_metacognition(candidates, evidence, attended.empty() ? 0.9L : 1.0L - attended.front().score,
                                                 config_.defer_threshold);
        result.confidence = meta.confidence;
        result.deferred = meta.should_defer;
        if (result.deferred) {
            result.answer = "insufficient evidence";
            if (!meta.known_gaps.empty()) {
                result.answer += ": " + meta.known_gaps.front();
            }
            result.saved_episode_id = memory_.record("now", std::string(prompt), result.answer, result.confidence, false, "defer");
        } else {
            result.answer = candidates.front();
            result.saved_episode_id = memory_.record("now", std::string(prompt), result.answer, result.confidence, true, "retrieval");
        }
        trace << "attention_top_k=" << attended.size() << "\n";
        trace << "working_memory=" << working_.slots().size() << "\n";
        trace << "confidence=" << static_cast<double>(result.confidence) << "\n";
        trace << "episode_id=" << result.saved_episode_id << "\n";
        result.trace = trace.str();
        return result;
    }

    const EpisodicMemory& memory() const noexcept {
        return memory_;
    }

    const SkillGenome& skills() const noexcept {
        return skills_;
    }

private:
    AgiFoundryConfig config_;
    EpisodicMemory memory_;
    WorkingMemory working_;
    AttentionField attention_;
    ProgramSynthesizer synthesizer_;
    DzetaVM vm_;
    SkillGenome skills_;
};

} // namespace dzeta
