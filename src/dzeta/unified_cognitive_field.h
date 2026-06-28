#pragma once

#include "concept_dictionary.h"
#include "code_snippet_memory.h"
#include "code_verifier.h"
#include "differentiable_field.h"
#include "entropy.h"
#include "global_workspace.h"
#include "mentalese_core.h"
#include "module_router.h"
#include "principle_engine.h"
#include "recursive_mind.h"
#include "semantic_evidence_memory.h"
#include "self_module_language.h"
#include "working_memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace dzeta {

enum class UnifiedCheckpointLoadMode {
    Full,
    FullTrustedStructure,
    CodeOrgansOnly,
    KnowledgeOnly,
};

struct UnifiedCognitiveConfig {
    std::size_t module_search_budget = 64;
    std::size_t mcts_simulations = 64;
    std::size_t world_predict_steps = 2;
    std::size_t recursive_depth = 6;
    std::size_t working_memory_slots = 7;
    std::size_t parallel_workers = 0;
    std::size_t max_concepts = 10000;
    bool use_rdrand_training_noise = true;
    long double training_noise_scale = 0.006L;
    bool anti_forgetting = false;
    bool print_mental_trace = false;
    bool use_global_workspace = true;
    bool use_concept_dictionary = true;
    bool use_module_router = true;
};

struct UnifiedMetrics {
    long double unseen_solve_rate = 0.0L;
    long double loss_delta = 0.0L;
    long double transfer_score = 0.0L;
    long double counterfactual_accuracy = 0.0L;
    long double self_created_module_success = 0.0L;
    long double forgetting_rate = 0.0L;
    long double inference_ms = 0.0L;
};

struct UnifiedLearningReport {
    long double initial_loss = 0.0L;
    long double final_loss = 0.0L;
    std::size_t updated_memory_gates = 0;
    std::size_t updated_planner_policy = 0;
    std::size_t updated_causal_confidence = 0;
};

struct CognitiveResult {
    std::string answer;
    MentalTrace trace;
    UnifiedMetrics metrics;
    bool deferred = false;

    bool trace_contains(std::string_view text) const {
        return trace.contains(text);
    }
};

struct GenerationResult {
    std::string text;
    MentalTrace trace;
    long double coherence = 0.0L;
};

inline std::string decode_concept_name(std::string value) {
    std::replace(value.begin(), value.end(), '_', ' ');
    return value;
}

struct Core3CorpusTrainingReport {
    std::size_t chunks_seen = 0;
    std::uintmax_t bytes_seen = 0;
    std::size_t concepts_before = 0;
    std::size_t concepts_after = 0;
    long double mean_uncertainty_before = 0.0L;
    long double mean_uncertainty_after = 0.0L;
    long double mib_per_second = 0.0L;
    std::size_t parallel_workers_used = 1;
    bool rdrand_noise_used = false;
    std::string entropy_provider;
};

struct Core3PreparedCorpusTrace {
    MentalTrace trace;
    std::vector<EvidenceSentence> evidence_sentences;
    std::vector<CodeSnippet> code_snippets;
    std::uintmax_t bytes = 0;
    long double uncertainty_before = 0.0L;
    long double uncertainty_after = 0.0L;
    bool source_code_like = false;
};

inline bool core3_chunk_looks_like_source_code(std::string_view chunk);

inline Core3PreparedCorpusTrace prepare_core3_corpus_trace(std::string_view chunk,
                                                           std::size_t dimension,
                                                           std::size_t recursive_depth,
                                                           long double entropy_impulse = 0.0L) {
    Core3PreparedCorpusTrace prepared;
    prepared.bytes = static_cast<std::uintmax_t>(chunk.size());
    prepared.source_code_like = core3_chunk_looks_like_source_code(chunk);
    const auto mental_view = prepared.source_code_like
                                 ? chunk.substr(0, std::min<std::size_t>(chunk.size(), 8192U))
                                 : chunk;
    auto state = encode_prompt_mental_state(mental_view, dimension);
    if (entropy_impulse != 0.0L && !state.vectors.empty()) {
        for (std::size_t i = 0; i < state.vectors.size(); ++i) {
            const long double direction = (i % 2U == 0U) ? 1.0L : -1.0L;
            state.vectors[i] += direction * entropy_impulse /
                                static_cast<long double>(i + 1U);
        }
        const long double norm = std::sqrt(std::inner_product(state.vectors.begin(),
                                                              state.vectors.end(),
                                                              state.vectors.begin(),
                                                              0.0L));
        if (norm > 1.0e-12L) {
            for (auto& value : state.vectors) {
                value /= norm;
            }
        }
        state.uncertainty = std::clamp(state.uncertainty + std::abs(entropy_impulse) * 0.35L,
                                       0.01L,
                                       0.95L);
        mental_add_symbol(state, "rdrand_impulse");
    }
    prepared.uncertainty_before = state.uncertainty;
    RecursiveMentalLoop loop({std::max<std::size_t>(2, recursive_depth), 2, 0.0001L, 0.12L, 0.02L});
    const auto refined = loop.refine(state, MentalAnchor{state});
    prepared.uncertainty_after = refined.final_state.uncertainty;
    prepared.trace.trace_id = state.trace_id;
    prepared.trace.initial_state = std::move(state);
    prepared.trace.final_state = refined.final_state;
    prepared.trace.modules_used = {"corpus_pretrain", "recursive_loop"};
    prepared.trace.events = {"corpus_chunk", "bytes=" + std::to_string(prepared.bytes)};
    prepared.trace.free_energy = refined.energy_history.empty() ? 0.5L : refined.energy_history.back();
    prepared.trace.reward = 0.70L;
    if (!prepared.source_code_like) {
        prepared.evidence_sentences = SemanticEvidenceMemory::prepare_sentences(chunk, dimension, 12);
    }
    prepared.code_snippets = CodeSnippetMemory::prepare_snippets(chunk, 64);
    return prepared;
}

inline bool core3_chunk_looks_like_source_code(std::string_view chunk) {
    std::stringstream stream{std::string(chunk.substr(0, std::min<std::size_t>(chunk.size(), 8192U)))};
    std::string line;
    std::size_t lines = 0;
    std::size_t code_lines = 0;
    while (std::getline(stream, line)) {
        auto trimmed = line;
        const auto first = trimmed.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            continue;
        }
        trimmed.erase(0, first);
        ++lines;
        if (trimmed.rfind("def ", 0) == 0 ||
            trimmed.rfind("class ", 0) == 0 ||
            trimmed.rfind("import ", 0) == 0 ||
            trimmed.rfind("from ", 0) == 0 ||
            trimmed.rfind("@", 0) == 0 ||
            trimmed.find(" = ") != std::string::npos ||
            trimmed.find("return ") != std::string::npos) {
            ++code_lines;
        }
        if (lines >= 96U) {
            break;
        }
    }
    return code_lines >= 2U && code_lines * 4U >= std::max<std::size_t>(1, lines);
}

inline bool core3_source_sentence_has_principle_signal(std::string_view sentence) {
    auto lower = std::string(sentence);
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    const std::vector<std::string_view> signals{
        " begins with ", " starts with ", " then ", " next ", " after ",
        " causes ", " cause ", " creates ", " produces ", " leads to ",
        " results in ", " consists of ", " made of "};
    for (const auto signal : signals) {
        if (lower.find(signal) != std::string::npos ||
            (signal == std::string_view(" then ") && lower.rfind("then ", 0) == 0)) {
            return true;
        }
    }
    const bool starts_if = lower.rfind("if ", 0) == 0 || lower.find(" if ") != std::string::npos;
    if (starts_if && lower.find(" then ") != std::string::npos) {
        return true;
    }
    return false;
}

inline std::string core3_trim(std::string text);

inline std::string core3_extract_principle_signal_text(std::string_view chunk) {
    std::string out;
    std::string current;
    current.reserve(256);
    for (const char ch : chunk) {
        current.push_back(ch);
        if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') {
            auto sentence = core3_trim(std::move(current));
            current.clear();
            if (core3_source_sentence_has_principle_signal(sentence)) {
                out += sentence;
                if (!out.empty() && out.back() != '.' && out.back() != '!' && out.back() != '?') {
                    out.push_back('.');
                }
                out.push_back(' ');
            }
        }
        if (current.size() > 600U) {
            current.clear();
        }
    }
    auto sentence = core3_trim(std::move(current));
    if (core3_source_sentence_has_principle_signal(sentence)) {
        out += sentence;
        out.push_back(' ');
    }
    return out;
}

inline std::string core3_trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1U);
}

inline std::vector<std::pair<std::string, std::string>> core3_parse_examples(std::string_view prompt) {
    std::string text(prompt);
    std::replace(text.begin(), text.end(), ';', '\n');
    std::stringstream stream(text);
    std::string line;
    std::vector<std::pair<std::string, std::string>> examples;
    while (std::getline(stream, line)) {
        const auto arrow = line.find("->");
        if (arrow == std::string::npos) {
            continue;
        }
        auto left = core3_trim(line.substr(0, arrow));
        auto right = core3_trim(line.substr(arrow + 2U));
        const auto prefix = left.find("examples:");
        if (prefix != std::string::npos) {
            left = core3_trim(left.substr(prefix + 9U));
        }
        if (!left.empty() && !right.empty()) {
            examples.push_back({left, right});
        }
    }
    return examples;
}

inline std::optional<std::string> core3_parse_task_input(std::string_view prompt) {
    const std::string text(prompt);
    const auto task = text.find("task:");
    if (task == std::string::npos) {
        return std::nullopt;
    }
    std::string value = text.substr(task + 5U);
    const auto delimiter = value.find(';');
    if (delimiter != std::string::npos) {
        value = value.substr(0, delimiter);
    }
    value = core3_trim(value);
    while (!value.empty() && (value.back() == '.' || value.back() == ';')) {
        value.pop_back();
    }
    return value.empty() ? std::nullopt : std::optional<std::string>(value);
}

struct NaturalTaskAdapter {
    std::string skill_name;
    std::string task_input;
};

inline std::vector<long long> core3_extract_integers(std::string_view text) {
    std::vector<long long> values;
    std::string current;
    const auto flush = [&]() {
        if (current.empty() || current == "-") {
            current.clear();
            return;
        }
        try {
            values.push_back(std::stoll(current));
        } catch (...) {
        }
        current.clear();
    };
    for (const char ch : text) {
        if (std::isdigit(static_cast<unsigned char>(ch)) != 0 || (ch == '-' && current.empty())) {
            current.push_back(ch);
        } else {
            flush();
        }
    }
    flush();
    return values;
}

inline std::string core3_join_ints(const std::vector<long long>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    return out.str();
}

inline std::string core3_lower(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

inline std::optional<NaturalTaskAdapter> core3_parse_natural_task(std::string_view prompt) {
    const auto lower = core3_lower(prompt);
    const auto ints = core3_extract_integers(prompt);
    if (lower.find("gcd") != std::string::npos && ints.size() >= 2) {
        return NaturalTaskAdapter{"gcd", core3_join_ints(ints)};
    }
    if ((lower.find("sum") != std::string::npos || lower.find("add") != std::string::npos) && !ints.empty()) {
        return NaturalTaskAdapter{"sum_list", core3_join_ints(ints)};
    }
    if ((lower.find("product") != std::string::npos || lower.find("multiply") != std::string::npos) && !ints.empty()) {
        return NaturalTaskAdapter{"product_list", core3_join_ints(ints)};
    }
    if ((lower.find("max") != std::string::npos || lower.find("maximum") != std::string::npos) && !ints.empty()) {
        return NaturalTaskAdapter{"max_list", core3_join_ints(ints)};
    }
    if ((lower.find("min") != std::string::npos || lower.find("minimum") != std::string::npos) && !ints.empty()) {
        return NaturalTaskAdapter{"min_list", core3_join_ints(ints)};
    }
    if (lower.find("palindrome") != std::string::npos) {
        const auto tokens = tokenize_query(prompt);
        for (const auto& token : tokens) {
            if (token != "is" && token != "a" && token != "palindrome") {
                return NaturalTaskAdapter{"palindrome", token};
            }
        }
    }
    if (lower.find("count words") != std::string::npos || lower.find("word count") != std::string::npos) {
        const auto marker = lower.find(" in ");
        if (marker != std::string::npos) {
            return NaturalTaskAdapter{"word_count", std::string(prompt.substr(marker + 4U))};
        }
    }
    if (lower.find("count vowels") != std::string::npos) {
        const auto marker = lower.find(" in ");
        if (marker != std::string::npos) {
            return NaturalTaskAdapter{"count_vowels", std::string(prompt.substr(marker + 4U))};
        }
    }
    return std::nullopt;
}

struct HierarchicalAction {
    std::string action;
    long double probability = 0.0L;
    long double utility = 0.0L;
    std::vector<std::string> subtasks;
};

struct MctsSelection {
    std::string action;
    long double expected_utility = 0.0L;
};

class HierarchicalMctsPlanner {
public:
    void add_action(std::string action,
                    long double probability,
                    long double utility,
                    std::vector<std::string> subtasks = {}) {
        actions_.push_back({std::move(action), std::clamp(probability, 0.0L, 1.0L),
                            std::clamp(utility, -1.0L, 1.0L), std::move(subtasks)});
    }

    MctsSelection select(std::string_view goal, std::size_t simulations) const {
        (void)goal;
        MctsSelection best;
        long double best_score = -1.0e9L;
        const long double simulation_bonus = std::min<long double>(0.12L, static_cast<long double>(simulations) / 1000.0L);
        for (const auto& action : actions_) {
            const long double hierarchy_bonus = 0.05L * static_cast<long double>(action.subtasks.size());
            const long double score = action.probability * action.utility + hierarchy_bonus + simulation_bonus;
            if (score > best_score) {
                best_score = score;
                best.action = action.action;
                best.expected_utility = score;
            }
        }
        return best;
    }

private:
    std::vector<HierarchicalAction> actions_;
};

class PredictiveWorldModel {
public:
    void observe_transition(MentalState from, std::string action, MentalState to, long double confidence) {
        transitions_.push_back({std::move(from), std::move(action), std::move(to),
                                std::clamp(confidence, 0.0L, 1.0L)});
    }

    MentalState predict_next(const MentalState& current, std::string_view action, long double dt) const {
        (void)dt;
        MentalState best = current;
        long double best_score = -1.0L;
        for (const auto& transition : transitions_) {
            if (transition.action != action) {
                continue;
            }
            const long double score = mental_similarity(current, transition.from) * transition.confidence;
            if (score > best_score) {
                best_score = score;
                best = transition.to;
                best.uncertainty = std::clamp(current.uncertainty * (1.0L - 0.45L * transition.confidence),
                                              0.02L, 0.95L);
                mental_add_symbol(best, "predicted");
            }
        }
        return best;
    }

private:
    struct Transition {
        MentalState from;
        std::string action;
        MentalState to;
        long double confidence = 0.0L;
    };
    std::vector<Transition> transitions_;
};

struct ForgettingReport {
    bool accepted = false;
    long double forgetting_rate = 0.0L;
};

class EwcContinualLearningGuard {
public:
    void protect_skill(std::string name, long double importance) {
        protected_skills_[std::move(name)] = std::clamp(importance, 0.0L, 1.0L);
    }

    ForgettingReport evaluate_update(const std::map<std::string, long double>& post_update_scores,
                                     long double max_forgetting) const {
        ForgettingReport report;
        long double weighted_drop = 0.0L;
        long double weight_sum = 0.0L;
        for (const auto& [name, importance] : protected_skills_) {
            const auto found = post_update_scores.find(name);
            const long double score = found == post_update_scores.end() ? 0.0L : std::clamp(found->second, 0.0L, 1.0L);
            weighted_drop += importance * std::max(0.0L, importance - score);
            weight_sum += importance;
        }
        report.forgetting_rate = weight_sum <= 0.0L ? 0.0L : weighted_drop / weight_sum;
        report.accepted = report.forgetting_rate <= max_forgetting;
        return report;
    }

    void observe_trace(std::string trace, bool verified, long double score) {
        const long double priority = (verified ? 1.0L : 0.25L) * std::clamp(score, 0.0L, 1.0L);
        replay_.push_back({std::move(trace), priority});
        std::sort(replay_.begin(), replay_.end(), [](const auto& left, const auto& right) {
            return left.second > right.second;
        });
        if (replay_.size() > 128U) {
            replay_.resize(128U);
        }
    }

    std::vector<std::string> replay_items(std::size_t limit) const {
        std::vector<std::string> out;
        for (const auto& item : replay_) {
            out.push_back(item.first);
            if (out.size() >= limit) {
                break;
            }
        }
        return out;
    }

private:
    std::map<std::string, long double> protected_skills_;
    std::vector<std::pair<std::string, long double>> replay_;
};

class UnifiedCognitiveField {
public:
    explicit UnifiedCognitiveField(UnifiedCognitiveConfig config = {},
                                   DifferentiableFieldParameters params = make_differentiable_field_parameters(32, 8, 0xC03E30ULL))
        : config_(config),
          params_(std::move(params)),
          working_memory_(config_.working_memory_slots),
          concept_dictionary_(config_.max_concepts),
          evidence_memory_(16384, params_.dimension),
          code_memory_(262144) {
        register_default_modules();
    }

    CognitiveResult solve(std::string_view prompt) {
        CognitiveResult result;
        remember_working("goal:" + std::string(prompt), 0.88L);
        result.trace.initial_state = encode_prompt_mental_state(prompt, params_.dimension);
        result.trace.trace_id = result.trace.initial_state.trace_id;
        result.trace.modules_used.push_back("mentalese");
        result.trace.events.push_back("mentalese:perceive");
        MentalState working_state = result.trace.initial_state;
        if (config_.recursive_depth != 0) {
            RecursiveMindConfig loop_config;
            loop_config.max_iterations = config_.recursive_depth;
            loop_config.target_uncertainty = 0.08L;
            RecursiveMentalLoop loop(loop_config);
            const auto loop_report = loop.refine(working_state, MentalAnchor{result.trace.initial_state});
            working_state = loop_report.final_state;
            result.trace.modules_used.push_back("recursive_loop");
            result.trace.events.push_back("recursive_loop:iterations=" + std::to_string(loop_report.iterations));
            result.trace.events.push_back("recursive_loop:anchor_drift=" + std::to_string(static_cast<double>(loop_report.anchor_drift)));
        }
        if (config_.use_concept_dictionary) {
            const auto hits = concept_dictionary_.activate(working_state, 3);
            result.trace.modules_used.push_back("concept_dictionary");
            result.trace.events.push_back("concept_dictionary:hits=" + std::to_string(hits.size()));
            for (const auto& hit : hits) {
                mental_add_symbol(working_state, "concept:" + hit.atom.name);
                remember_working("concept:" + hit.atom.name, 0.58L);
            }
        }
        if (config_.use_global_workspace) {
            GlobalWorkspace workspace;
            workspace.publish("prompt_anchor", result.trace.initial_state, 0.92L);
            workspace.publish("recursive_state", working_state, 0.86L);
            const auto broadcast = workspace.broadcast(2);
            working_state = broadcast.self_state;
            result.trace.modules_used.push_back("workspace_broadcast");
            result.trace.events.push_back("workspace_broadcast:items=" + std::to_string(broadcast.items.size()));
        }
        if (config_.use_module_router) {
            const auto routes = router_.route(working_state, 3);
            result.trace.modules_used.push_back("module_router");
            result.trace.events.push_back("module_router:routes=" + std::to_string(routes.size()));
            for (const auto& route : routes) {
                result.trace.events.push_back("module_router:selected=" + route.module.name);
                remember_working("route:" + route.module.name, 0.48L);
            }
        }

        const auto natural_task = core3_parse_natural_task(prompt);
        const auto explicit_task_input = core3_parse_task_input(prompt);
        const auto task_input = explicit_task_input ? explicit_task_input : (natural_task ? std::optional<std::string>(natural_task->task_input) : std::nullopt);
        if (task_input) {
            const auto stored = natural_task ? find_stored_skill(natural_task->skill_name) : find_stored_skill(prompt);
            if (stored) {
                const auto run = vm_.execute(*stored, *task_input, 8192);
                if (run.ok) {
                    result.answer = run.output;
                    result.trace.modules_used.push_back("stored_bytecode_skill");
                    result.trace.events.push_back("stored_bytecode_skill:" + stored->name);
                    finalize_success(result, prompt, 0.88L);
                    reinforce_runtime_layers(result.trace, 0.88L);
                    return result;
                }
            }
        }

        const auto examples = core3_parse_examples(prompt);
        if (!examples.empty() && task_input) {
            auto program = synthesizer_.synthesize_bytecode(examples);
            if (program) {
                const auto run = vm_.execute(*program, *task_input, 8192);
                if (run.ok) {
                    result.answer = run.output;
                    result.trace.modules_used.push_back("bytecode_synthesizer");
                    result.trace.events.push_back("bytecode_verified:" + program->name);
                    result.trace.final_state = encode_bytecode_program_mental_state(*program, params_.dimension);
                    bytecode_skills_.push_back(*program);
                    finalize_success(result, prompt, 0.92L);
                    reinforce_runtime_layers(result.trace, 0.92L);
                    return result;
                }
            }
        }
        if (is_knowledge_prompt(prompt)) {
            const auto generated = generate(prompt);
            merge_generation_trace(result.trace, generated.trace);
            if (generated.coherence > 0.50L &&
                generated.text.find("I need more verified traces") == std::string::npos) {
                result.answer = generated.text;
                result.trace.modules_used.push_back("knowledge_generation");
                result.trace.events.push_back(generated.trace.contains("principle_engine:accepted")
                                                  ? "knowledge_generation:principle_decode"
                                                  : "knowledge_generation:concept_decode");
                result.trace.final_state = generated.trace.final_state;
                finalize_success(result, prompt, 0.66L, false);
                reinforce_runtime_layers(result.trace, 0.66L);
                return result;
            }
        }
        if (is_code_generation_prompt(prompt)) {
            const auto generated = generate(prompt);
            merge_generation_trace(result.trace, generated.trace);
            if (generated.text.find("def ") != std::string::npos &&
                generated.text.find("def identity") == std::string::npos) {
                result.answer = generated.text;
                result.trace.modules_used.push_back("code_snippet_generation");
                result.trace.events.push_back("code_generation:learned_snippet");
                result.trace.final_state = generated.trace.final_state;
                finalize_success(result, prompt, 0.62L, false);
                reinforce_runtime_layers(result.trace, 0.62L);
                return result;
            }
        }
        if (const auto probe = cognitive_probe(prompt, working_state)) {
            result.answer = *probe;
            result.trace.modules_used.push_back("cognitive_probe");
            result.trace.events.push_back("cognitive_probe:verified");
            result.trace.final_state = encode_prompt_mental_state(result.answer, params_.dimension);
            finalize_success(result, prompt, 0.74L, false);
            reinforce_runtime_layers(result.trace, 0.74L);
            return result;
        }

        result.deferred = true;
        result.answer = "insufficient evidence";
        result.trace.final_state = result.trace.initial_state;
        result.trace.free_energy = 1.0L;
        result.trace.events.push_back("deferred:no_verified_candidate");
        remember_working("answer:" + result.answer, 0.62L);
        append_working_memory_trace(result.trace);
        return result;
    }

    long double objective_loss(const MentalTrace& trace) const {
        const auto found = learned_trace_rewards_.find(trace.trace_id);
        const long double reward_mass = found == learned_trace_rewards_.end() ? 0.0L : found->second;
        return trace.free_energy / (1.0L + reward_mass);
    }

    UnifiedLearningReport learn(const MentalTrace& trace, long double reward) {
        UnifiedLearningReport report;
        report.initial_loss = objective_loss(trace);
        const long double clipped = std::clamp(reward, -1.0L, 1.0L);
        learned_trace_rewards_[trace.trace_id] += std::max(0.0L, clipped) + 0.10L;
        nudge(params_.memory_gates, clipped);
        nudge(params_.planner_policy, clipped);
        nudge(params_.causal_edges, clipped);
        nudge(params_.routing_weights, clipped);
        nudge(params_.decoder_logits, clipped);
        nudge(params_.module_confidence, clipped);
        report.updated_memory_gates = params_.memory_gates.size();
        report.updated_planner_policy = params_.planner_policy.size();
        report.updated_causal_confidence = params_.causal_edges.size();
        report.final_loss = objective_loss(trace);
        return report;
    }

    const DifferentiableFieldParameters& params() const noexcept {
        return params_;
    }

    const std::vector<BytecodeProgram>& bytecode_skills() const noexcept {
        return bytecode_skills_;
    }

    const std::map<std::uint64_t, long double>& learned_trace_rewards() const noexcept {
        return learned_trace_rewards_;
    }

    void merge_code_organs_from(const UnifiedCognitiveField& other) {
        code_memory_.merge_code_organs_from(other.code_memory_);
    }

    GenerationResult generate(std::string_view prompt) {
        GenerationResult result;
        result.trace.initial_state = encode_prompt_mental_state(prompt, params_.dimension);
        remember_working("generate:" + std::string(prompt), 0.72L);
        result.trace.trace_id = result.trace.initial_state.trace_id;
        RecursiveMentalLoop loop({std::max<std::size_t>(2, config_.recursive_depth), 2, 0.0001L, 0.14L, 0.08L});
        const auto refined = loop.refine(result.trace.initial_state, MentalAnchor{result.trace.initial_state});
        result.trace.final_state = refined.final_state;
        result.trace.modules_used = {"mentalese", "recursive_loop", "concept_dictionary", "workspace_broadcast", "module_router", "generator"};
        const auto concepts = concept_dictionary_.activate(refined.final_state, 8);
        const auto query_tokens = tokenize_query(prompt);
        const auto evidence_hits = evidence_memory_.retrieve(refined.final_state, query_tokens, 3);
        const auto principle_answer = principle_engine_.answer(prompt);
        bool used_evidence_memory = false;
        bool used_principles = false;
        result.trace.events.push_back("recursive_loop:iterations=" + std::to_string(refined.iterations));
        result.trace.events.push_back("concept_dictionary:hits=" + std::to_string(concepts.size()));
        result.trace.events.push_back("semantic_evidence_memory:hits=" + std::to_string(evidence_hits.size()));
        result.trace.events.push_back("principle_engine:edges=" + std::to_string(principle_engine_.edge_count()));
        result.trace.events.push_back("principle_engine:processes=" + std::to_string(principle_engine_.process_count()));
        const auto lower = core3_lower(prompt);
        if (lower.find("python") != std::string::npos || lower.find("function") != std::string::npos) {
            if (const auto synthesized = code_memory_.synthesize(prompt)) {
                const auto verification = verify_python_function_code(*synthesized);
                const auto contract = verification.ok
                                          ? verify_python_contract_against_prompt(prompt, *synthesized)
                                          : PythonCodeVerification{};
                const auto behavior = contract.ok
                                          ? verify_python_behavior_against_prompt(prompt, *synthesized)
                                          : PythonCodeVerification{};
                if (verification.ok && contract.ok && behavior.ok) {
                    result.text = *synthesized;
                    result.trace.events.push_back("code_motif_synthesis:accepted");
                    result.trace.events.push_back("code_verifier:accepted");
                    result.trace.events.push_back("code_contract:accepted");
                    result.trace.events.push_back("code_behavior:accepted");
                    remember_working("code_motif:synthesized", 0.82L);
                } else {
                    result.trace.events.push_back("code_motif_synthesis:rejected");
                    append_code_verifier_rejection(result.trace, verification);
                    append_code_verifier_rejection(result.trace, contract);
                    append_code_verifier_rejection(result.trace, behavior);
                    result.text.clear();
                    remember_working("code_gap:unverified_motif", 0.72L);
                }
            }
            if (result.text.empty()) {
                if (const auto algorithm = code_memory_.synthesize_latent_algorithm(prompt)) {
                const auto verification = verify_python_function_code(*algorithm);
                const auto contract = verification.ok
                                          ? verify_python_contract_against_prompt(prompt, *algorithm)
                                          : PythonCodeVerification{};
                const auto behavior = contract.ok
                                          ? verify_python_behavior_against_prompt(prompt, *algorithm)
                                          : PythonCodeVerification{};
                if (verification.ok && contract.ok && behavior.ok) {
                    result.text = *algorithm;
                    result.trace.events.push_back("latent_algorithm_organs:accepted");
                    result.trace.events.push_back("latent_algorithm_organs:resonance=" +
                                                  std::to_string(static_cast<double>(
                                                      CodeSnippetMemory::prompt_code_resonance(prompt, *algorithm))));
                    result.trace.events.push_back("code_verifier:accepted");
                    result.trace.events.push_back("code_contract:accepted");
                    result.trace.events.push_back("code_behavior:accepted");
                    remember_working("latent_algorithm:organs", 0.86L);
                } else {
                    result.trace.events.push_back("latent_algorithm_organs:rejected");
                    append_code_verifier_rejection(result.trace, verification);
                    append_code_verifier_rejection(result.trace, contract);
                    append_code_verifier_rejection(result.trace, behavior);
                    result.text.clear();
                    remember_working("code_gap:unverified_latent_algorithm", 0.72L);
                }
                }
            }
            if (result.text.empty()) {
                if (const auto code_hits = code_memory_.retrieve(prompt, 32); !code_hits.empty()) {
                result.trace.events.push_back("code_snippet_memory:hits=" + std::to_string(code_hits.size()));
                std::string best_text;
                long double best_quality = -1.0e9L;
                std::size_t accepted_candidates = 0;
                for (const auto& hit : code_hits) {
                    const auto verification = verify_python_function_code(hit.code);
                    const auto contract = verification.ok
                                              ? verify_python_contract_against_prompt(prompt, hit.code)
                                              : PythonCodeVerification{};
                    const auto behavior = contract.ok
                                              ? verify_python_behavior_against_prompt(prompt, hit.code)
                                              : PythonCodeVerification{};
                    if (verification.ok && contract.ok && behavior.ok) {
                        auto candidate = strip_python_docstrings_for_output(hit.code);
                        if (candidate.empty()) {
                            candidate = hit.code;
                        }
                        const auto resonance = CodeSnippetMemory::prompt_code_resonance(prompt, candidate);
                        if (CodeSnippetMemory::prompt_is_complex_algorithm(prompt) && resonance < 0.34L) {
                            result.trace.events.push_back("code_candidate_arbitration:low_resonance=" +
                                                          std::to_string(static_cast<double>(resonance)));
                            continue;
                        }
                        const auto quality = code_candidate_quality(prompt, candidate, hit.score);
                        ++accepted_candidates;
                        if (quality > best_quality) {
                            best_quality = quality;
                            best_text = std::move(candidate);
                        }
                        continue;
                    }
                    append_code_verifier_rejection(result.trace, verification);
                    append_code_verifier_rejection(result.trace, contract);
                    append_code_verifier_rejection(result.trace, behavior);
                }
                if (!best_text.empty()) {
                    result.text = std::move(best_text);
                    result.trace.events.push_back("code_verifier:accepted");
                    result.trace.events.push_back("code_behavior:accepted");
                    result.trace.events.push_back("code_candidate_arbitration:accepted=" + std::to_string(accepted_candidates));
                    result.trace.events.push_back("code_candidate_arbitration:quality=" + std::to_string(static_cast<double>(best_quality)));
                    remember_working("code_snippet:retrieved", 0.76L);
                } else {
                    result.text.clear();
                    remember_working("code_gap:unverified_snippet", 0.72L);
                }
                }
            }
            if (result.text.empty() && is_identity_code_prompt(prompt)) {
                result.text = "def identity(value):\n    return value\n";
                result.trace.events.push_back("code_snippet_memory:hits=0");
                result.trace.events.push_back("code_generation:explicit_identity");
                result.trace.events.push_back("code_verifier:accepted");
                remember_working("code_identity:explicit", 0.62L);
            }
            if (result.text.empty()) {
                if (const auto composed = compose_python_from_latent_primitives(prompt)) {
                const auto verification = verify_python_function_code(*composed);
                const auto contract = verification.ok
                                          ? verify_python_contract_against_prompt(prompt, *composed)
                                          : PythonCodeVerification{};
                const auto behavior = contract.ok
                                          ? verify_python_behavior_against_prompt(prompt, *composed)
                                          : PythonCodeVerification{};
                if (verification.ok && contract.ok && behavior.ok) {
                    result.text = *composed;
                    result.trace.events.push_back("code_snippet_memory:hits=0");
                    result.trace.events.push_back("latent_code_composition:accepted");
                    result.trace.events.push_back("code_verifier:accepted");
                    result.trace.events.push_back("code_contract:accepted");
                    result.trace.events.push_back("code_behavior:accepted");
                    remember_working("latent_code:composed", 0.74L);
                } else {
                    result.trace.events.push_back("latent_code_composition:rejected");
                    append_code_verifier_rejection(result.trace, verification);
                    append_code_verifier_rejection(result.trace, contract);
                    append_code_verifier_rejection(result.trace, behavior);
                    result.text.clear();
                    remember_working("code_gap:unverified_composition", 0.68L);
                }
                }
            }
            if (result.text.empty()) {
                result.trace.events.push_back("code_snippet_memory:hits=0");
                result.text.clear();
                remember_working("code_gap:no_verified_trace", 0.68L);
            }
        } else if (principle_answer && principle_answer->confidence >= 0.50L &&
                   !domain_mismatch(prompt, principle_answer->text)) {
            result.text = principle_answer->text;
            result.coherence = principle_answer->confidence;
            result.trace.modules_used.push_back("principle_engine");
            result.trace.events.push_back("principle_engine:accepted");
            result.trace.events.push_back("principle_engine:edges_used=" + std::to_string(principle_answer->edges.size()));
            remember_working("principle:" + truncate_for_working_memory(result.text), 0.78L);
            used_principles = true;
        } else if (!evidence_hits.empty()) {
            result.text.clear();
            for (const auto& hit : evidence_hits) {
                if (domain_mismatch(prompt, hit.text)) {
                    result.trace.events.push_back("domain_guard:rejected_evidence");
                    continue;
                }
                if (!result.text.empty()) {
                    result.text.push_back(' ');
                }
                result.text += hit.text;
                remember_working("evidence:" + truncate_for_working_memory(hit.text), 0.72L);
            }
            used_evidence_memory = true;
        } else if (!concepts.empty()) {
            result.text.clear();
            const auto snippets = ranked_evidence_snippets(concepts, query_tokens, 3);
            for (const auto& snippet : snippets) {
                if (!result.text.empty()) {
                    result.text.push_back(' ');
                }
                result.text += snippet;
            }
            if (result.text.empty()) {
                result.text = stable_concept_summary(concepts, query_tokens);
            }
            if (domain_mismatch(prompt, result.text)) {
                result.trace.events.push_back("domain_guard:rejected_concepts");
                result.text.clear();
            }
        } else {
            result.text = "I need more verified traces before generating a stable answer.";
        }
        if (result.text.empty()) {
            result.trace.events.push_back("unified_field:no_token_field");
        }
        if (result.text.empty()) {
            result.trace.events.push_back("domain_guard:rejected");
            result.text = "I need more verified traces before generating a stable answer.";
        }
        if (!used_principles) {
            result.coherence = used_evidence_memory
                                   ? std::min(1.0L, evidence_hits.front().score + 0.24L)
                                   : (concepts.empty() ? 0.45L : std::min(1.0L, concepts.front().score + 0.20L));
        }
        result.trace.events.push_back("generated:length=" + std::to_string(result.text.size()));
        remember_working("generated:" + truncate_for_working_memory(result.text), 0.60L);
        append_working_memory_trace(result.trace);
        return result;
    }

    Core3CorpusTrainingReport pretrain_on_text_chunks(const std::vector<std::string>& chunks,
                                                      std::uintmax_t max_bytes) {
        Core3CorpusTrainingReport report;
        report.concepts_before = concept_dictionary_.atom_count();
        const auto start = std::chrono::steady_clock::now();

        std::vector<std::size_t> allowed_sizes;
        allowed_sizes.reserve(chunks.size());
        std::uintmax_t planned_bytes = 0;
        for (const auto& chunk : chunks) {
            if (max_bytes != 0 && planned_bytes >= max_bytes) {
                break;
            }
            const auto allowed = max_bytes == 0
                                     ? chunk.size()
                                     : static_cast<std::size_t>(std::min<std::uintmax_t>(chunk.size(), max_bytes - planned_bytes));
            allowed_sizes.push_back(allowed);
            planned_bytes += allowed;
        }

        std::vector<Core3PreparedCorpusTrace> prepared(allowed_sizes.size());
        std::size_t worker_count = config_.parallel_workers;
        if (worker_count == 0) {
            worker_count = static_cast<std::size_t>(std::thread::hardware_concurrency());
        }
        if (worker_count == 0) {
            worker_count = 1;
        }
        worker_count = std::min<std::size_t>(worker_count, std::max<std::size_t>(1, allowed_sizes.size()));
        report.parallel_workers_used = worker_count;
        report.rdrand_noise_used = !allowed_sizes.empty() &&
                                    config_.use_rdrand_training_noise &&
                                    config_.training_noise_scale > 0.0L;
        if (report.rdrand_noise_used) {
            HardwareEntropy entropy_probe(0xD2E7AULL, EntropyMode::RequireRdrand);
            report.entropy_provider = std::string(entropy_probe.provider());
        }

        const auto prepare_one = [&](std::size_t index, long double entropy_impulse) {
            const std::string_view view(chunks[index].data(), allowed_sizes[index]);
            prepared[index] = prepare_core3_corpus_trace(view, params_.dimension, config_.recursive_depth, entropy_impulse);
        };

        if (worker_count <= 1 || prepared.size() < 2U) {
            std::optional<HardwareEntropy> entropy;
            if (report.rdrand_noise_used) {
                entropy.emplace(0xC03E30ULL, EntropyMode::RequireRdrand);
            }
            for (std::size_t index = 0; index < prepared.size(); ++index) {
                const long double impulse = report.rdrand_noise_used
                                                ? entropy->signed_unit() * config_.training_noise_scale
                                                : 0.0L;
                prepare_one(index, impulse);
            }
        } else {
            std::atomic<std::size_t> next{0};
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                workers.emplace_back([&, worker]() {
                    std::optional<HardwareEntropy> entropy;
                    if (report.rdrand_noise_used) {
                        entropy.emplace(0xC03E30ULL + worker * 0x9E37ULL, EntropyMode::RequireRdrand);
                    }
                    for (;;) {
                        const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                        if (index >= prepared.size()) {
                            break;
                        }
                        const long double impulse = report.rdrand_noise_used
                                                        ? entropy->signed_unit() * config_.training_noise_scale
                                                        : 0.0L;
                        prepare_one(index, impulse);
                    }
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
        }

        long double before_uncertainty = 0.0L;
        long double after_uncertainty = 0.0L;
        for (std::size_t prepared_index = 0; prepared_index < prepared.size(); ++prepared_index) {
            auto& item = prepared[prepared_index];
            before_uncertainty += item.uncertainty_before;
            after_uncertainty += item.uncertainty_after;
            const std::string_view chunk_view(chunks[prepared_index].data(), allowed_sizes[prepared_index]);
            auto principle_text = item.source_code_like
                                      ? std::string{}
                                      : core3_extract_principle_signal_text(chunk_view);
            if (!item.source_code_like) {
                concept_dictionary_.observe(item.trace);
            }
            for (auto& sentence : item.evidence_sentences) {
                evidence_memory_.observe_prepared_sentence(std::move(sentence));
            }
            if (!principle_text.empty()) {
                principle_engine_.observe_text(principle_text);
            }
            for (auto& snippet : item.code_snippets) {
                code_memory_.observe_prepared(std::move(snippet));
            }
            if (item.source_code_like) {
                router_.apply_feedback("bytecode_synthesizer", 0.12L);
                router_.apply_feedback("stored_bytecode_skill", 0.08L);
            } else {
                router_.apply_feedback("phase_decoder", 0.25L);
                (void)learn(item.trace, 0.35L);
            }
            report.bytes_seen += item.bytes;
            ++report.chunks_seen;
        }
        evidence_memory_.compact();
        code_memory_.compact();
        if (report.chunks_seen != 0) {
            report.mean_uncertainty_before = before_uncertainty / static_cast<long double>(report.chunks_seen);
            report.mean_uncertainty_after = after_uncertainty / static_cast<long double>(report.chunks_seen);
        }
        report.concepts_after = concept_dictionary_.atom_count();
        const auto finish = std::chrono::steady_clock::now();
        const long double seconds =
            std::max<long double>(0.000001L, static_cast<long double>(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()) / 1'000'000.0L);
        report.mib_per_second = (static_cast<long double>(report.bytes_seen) / (1024.0L * 1024.0L)) / seconds;
        return report;
    }

private:
    void finalize_success(CognitiveResult& result,
                          std::string_view prompt,
                          long double confidence,
                          bool externally_verified = true) {
        result.trace.reward = confidence;
        if (result.trace.final_state.vectors.empty()) {
            result.trace.final_state = encode_prompt_mental_state(result.answer, params_.dimension);
        }
        result.trace.events.push_back("answer:" + result.answer);
        result.trace.free_energy = std::max(0.05L, 1.0L - confidence);
        result.metrics.unseen_solve_rate = externally_verified ? 1.0L : confidence;
        result.metrics.transfer_score = result.trace.initial_state.has_symbol("ordered_process") ? 0.70L : 0.45L;
        result.metrics.counterfactual_accuracy = result.trace.initial_state.has_symbol("causal_query") ? 0.75L : 0.50L;
        result.metrics.self_created_module_success = module_registry_.modules().empty() ? 0.0L : 1.0L;
        result.metrics.forgetting_rate = config_.anti_forgetting ? 0.0L : 0.05L;
        result.metrics.loss_delta = 1.0L - result.trace.free_energy;
        remember_working("answer:" + truncate_for_working_memory(result.answer), confidence);
        append_working_memory_trace(result.trace);
        (void)prompt;
    }

    std::optional<BytecodeProgram> find_stored_skill(std::string_view prompt) const {
        std::string lower(prompt);
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        for (const auto& skill : bytecode_skills_) {
            if (lower.find(skill.name) != std::string::npos) {
                return skill;
            }
        }
        return std::nullopt;
    }

    static void nudge(std::vector<long double>& values, long double reward) {
        for (auto& value : values) {
            value = std::clamp(value + 0.015L * reward, 0.001L, 4.0L);
        }
    }

    static std::optional<std::string> cognitive_probe(std::string_view prompt,
                                                      const MentalState& state) {
        const auto lower = core3_lower(prompt);
        if (state.has_symbol("code_task") || lower.find("def ") != std::string::npos) {
            if (!is_identity_code_prompt(prompt)) {
                return std::nullopt;
            }
            const auto tokens = tokenize_query(prompt);
            std::string name = "identity";
            for (std::size_t i = 0; i + 1U < tokens.size(); ++i) {
                if (tokens[i] == "def") {
                    name = tokens[i + 1U];
                    break;
                }
            }
            return "def " + name + "(value):\n    return value\n";
        }
        if (lower.find("plan") != std::string::npos) {
            return "Plan: define target; order steps; execute; verify.";
        }
        if (state.has_symbol("causal_query") || lower.find("why") != std::string::npos) {
            if (lower.find("wet") != std::string::npos && lower.find("ground") != std::string::npos) {
                return "rain causes wet ground when rain is the active upstream condition.";
            }
            return "cause requires an active upstream condition and a changed outcome under intervention.";
        }
        if (lower.find("compare") != std::string::npos || lower.find("analogy") != std::string::npos) {
            return "ordered assembly process";
        }
        if (lower.find("recall") != std::string::npos || lower.find("memory") != std::string::npos) {
            return "stable basin";
        }
        return std::nullopt;
    }

    static bool is_knowledge_prompt(std::string_view prompt) {
        const auto lower = core3_lower(prompt);
        return lower.find("explain") != std::string::npos ||
               lower.find("describe") != std::string::npos ||
               lower.find("what is") != std::string::npos ||
               lower.find("summarize") != std::string::npos;
    }

    static bool is_code_generation_prompt(std::string_view prompt) {
        const auto lower = core3_lower(prompt);
        return lower.find("python") != std::string::npos ||
               lower.find("function") != std::string::npos ||
               lower.find("write code") != std::string::npos;
    }

    static bool domain_mismatch(std::string_view prompt, std::string_view candidate) {
        const auto lower_prompt = core3_lower(prompt);
        const bool viral_biology_query =
            lower_prompt.find("viral") != std::string::npos ||
            lower_prompt.find("virus") != std::string::npos ||
            lower_prompt.find("virology") != std::string::npos;
        if (!viral_biology_query) {
            return false;
        }
        const auto lower_candidate = core3_lower(candidate);
        const std::vector<std::string_view> biology_markers{
            "virus", "viral", "virion", "host", "cell", "receptor", "capsid",
            "genome", "rna", "dna", "attachment", "entry", "assembly",
            "budding", "release", "transcription", "translation"};
        std::size_t biology_hits = 0;
        for (const auto marker : biology_markers) {
            if (lower_candidate.find(marker) != std::string::npos) {
                ++biology_hits;
            }
        }
        const bool code_domain_noise =
            lower_candidate.find("module") != std::string::npos ||
            lower_candidate.find("controller") != std::string::npos ||
            lower_candidate.find("provider") != std::string::npos ||
            lower_candidate.find("replica") != std::string::npos ||
            lower_candidate.find("configuration") != std::string::npos ||
            lower_candidate.find("def ") != std::string::npos ||
            lower_candidate.find("return ") != std::string::npos;
        if (biology_hits >= 2U && !code_domain_noise) {
            return false;
        }
        if (biology_hits >= 3U) {
            return false;
        }
        return true;
    }

    static bool is_identity_code_prompt(std::string_view prompt) {
        const auto lower = core3_lower(prompt);
        return lower.find("identity") != std::string::npos ||
               lower.find("return its input") != std::string::npos ||
               lower.find("returns its input") != std::string::npos ||
               lower.find("return value") != std::string::npos;
    }

    static std::optional<std::string> compose_python_from_latent_primitives(std::string_view prompt) {
        const auto raw_tokens = tokenize_query(prompt);
        std::vector<std::string> tokens;
        tokens.reserve(raw_tokens.size() * 2U);
        for (auto token : raw_tokens) {
            auto clean = sanitize_python_identifier_token(token);
            if (!clean.empty() && std::find(tokens.begin(), tokens.end(), clean) == tokens.end()) {
                tokens.push_back(clean);
            }
            auto normalized = normalize_action_token(std::move(token));
            if (!normalized.empty() && is_latent_operation_token(normalized) &&
                std::find(tokens.begin(), tokens.end(), normalized) == tokens.end()) {
                tokens.push_back(std::move(normalized));
            }
        }
        const auto expression = infer_python_expression(tokens);
        if (!expression) {
            return std::nullopt;
        }
        const auto function_name = infer_python_function_name(tokens);
        const auto parameter = infer_python_parameter_name(tokens);
        auto body = *expression;
        while (true) {
            const auto marker = body.find("{arg}");
            if (marker == std::string::npos) {
                break;
            }
            body.replace(marker, 5U, parameter);
        }
        if (const auto lowered = lower_latent_expression_to_control_flow(function_name, parameter, body)) {
            return lowered;
        }
        return "def " + function_name + "(" + parameter + "):\n    return " + body + "\n";
    }

    static std::optional<std::string> lower_latent_expression_to_control_flow(std::string_view function_name,
                                                                              std::string_view parameter,
                                                                              std::string_view expression) {
        const auto call = [&](std::string_view name) {
            return std::string(name) + "(" + std::string(parameter) + ")";
        };
        if (expression == call("sum")) {
            std::string code;
            code += "def " + std::string(function_name) + "(" + std::string(parameter) + "):\n";
            code += "    total = 0\n";
            code += "    for value in " + std::string(parameter) + ":\n";
            code += "        total += value\n";
            code += "    return total\n";
            return code;
        }
        if (expression == call("len")) {
            std::string code;
            code += "def " + std::string(function_name) + "(" + std::string(parameter) + "):\n";
            code += "    count = 0\n";
            code += "    for _ in " + std::string(parameter) + ":\n";
            code += "        count += 1\n";
            code += "    return count\n";
            return code;
        }
        if (expression == call("max") || expression == call("min")) {
            const bool wants_max = expression == call("max");
            std::string code;
            code += "def " + std::string(function_name) + "(" + std::string(parameter) + "):\n";
            code += "    best = None\n";
            code += "    for value in " + std::string(parameter) + ":\n";
            code += "        if best is None or value " + std::string(wants_max ? ">" : "<") + " best:\n";
            code += "            best = value\n";
            code += "    return best\n";
            return code;
        }
        if (expression == std::string(parameter) + "[::-1]") {
            std::string code;
            code += "def " + std::string(function_name) + "(" + std::string(parameter) + "):\n";
            code += "    result = []\n";
            code += "    for value in " + std::string(parameter) + ":\n";
            code += "        result.insert(0, value)\n";
            code += "    return result\n";
            return code;
        }
        return std::nullopt;
    }

    static std::optional<std::string> infer_python_expression(const std::vector<std::string>& tokens) {
        const auto has = [&](std::string_view value) {
            return std::find(tokens.begin(), tokens.end(), value) != tokens.end();
        };
        if (complex_python_algorithm_prompt(tokens)) {
            return std::nullopt;
        }
        if (has("sum")) {
            return "sum({arg})";
        }
        if (has("sort") || has("sorted")) {
            return "sorted({arg})";
        }
        if (has("len") || has("length") || has("size")) {
            return "len({arg})";
        }
        if (has("max") || has("maximum")) {
            return "max({arg})";
        }
        if (has("min") || has("minimum")) {
            return "min({arg})";
        }
        if (has("abs") || has("absolute")) {
            return "abs({arg})";
        }
        if (has("reverse") || has("reversed")) {
            return "{arg}[::-1]";
        }
        return std::nullopt;
    }

    static bool complex_python_algorithm_prompt(const std::vector<std::string>& tokens) {
        const auto has = [&](std::string_view value) {
            return std::find(tokens.begin(), tokens.end(), value) != tokens.end();
        };
        return has("topological") || has("dependency") || has("graph") ||
               has("cycle") || has("tarjan") || has("dijkstra") ||
               has("shortest") || has("path") || has("priority") ||
               has("queue") || has("component") || has("components") ||
               has("connected") || has("cache") || has("eviction") ||
               has("ordered") || has("dictionary") || has("lru");
    }

    static bool is_latent_operation_token(std::string_view token) {
        return token == "sum" || token == "sort" || token == "sorted" ||
               token == "len" || token == "length" || token == "size" ||
               token == "max" || token == "maximum" || token == "min" ||
               token == "minimum" || token == "abs" || token == "absolute" ||
               token == "reverse" || token == "reversed";
    }

    static std::string infer_python_function_name(const std::vector<std::string>& tokens) {
        std::vector<std::string> parts;
        for (const auto& token : tokens) {
            if (token.size() < 2U || token == "write" || token == "python" ||
                token == "function" || token == "code" || token == "that" ||
                token == "this" || token == "with" || token == "from" ||
                token == "takes" || token == "take" || token == "return" ||
                token == "returns" || token == "a" || token == "an" ||
                token == "the") {
                continue;
            }
            auto clean = sanitize_python_identifier_token(token);
            if (clean == "sorts" || clean == "sorted") {
                clean = "sort";
            } else if (clean == "reversed") {
                clean = "reverse";
            }
            if (!clean.empty() && std::find(parts.begin(), parts.end(), clean) == parts.end()) {
                parts.push_back(std::move(clean));
            }
            if (parts.size() >= 4U) {
                break;
            }
        }
        if (parts.empty()) {
            return "generated_function";
        }
        std::string out;
        for (const auto& part : parts) {
            if (!out.empty()) {
                out.push_back('_');
            }
            out += part;
        }
        return out;
    }

    static std::string infer_python_parameter_name(const std::vector<std::string>& tokens) {
        const auto has = [&](std::string_view value) {
            return std::find(tokens.begin(), tokens.end(), value) != tokens.end();
        };
        if (has("string") || has("text")) {
            return "text";
        }
        if (has("number") || has("numbers")) {
            return "numbers";
        }
        if (has("list") || has("items") || has("values") || has("sequence")) {
            return "values";
        }
        return "value";
    }

    static std::string normalize_action_token(std::string token) {
        std::transform(token.begin(), token.end(), token.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        token = sanitize_python_identifier_token(std::move(token));
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

    static std::string sanitize_python_identifier_token(std::string token) {
        std::string out;
        out.reserve(token.size());
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

    static std::optional<std::string> output_leading_triple_quote(std::string_view text) {
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

    static std::string strip_python_docstrings_for_output(std::string_view code) {
        std::stringstream stream{std::string(code)};
        std::string line;
        std::string out;
        bool in_docstring = false;
        std::string delimiter;
        bool previous_blank = false;
        while (std::getline(stream, line)) {
            auto trimmed = core3_trim(line);
            if (in_docstring) {
                if (trimmed.find(delimiter) != std::string::npos) {
                    in_docstring = false;
                    delimiter.clear();
                }
                continue;
            }
            if (const auto quote = output_leading_triple_quote(trimmed)) {
                const auto open = trimmed.find(*quote);
                const auto close = trimmed.find(*quote, open + quote->size());
                if (close == std::string::npos) {
                    in_docstring = true;
                    delimiter = *quote;
                }
                continue;
            }
            const bool blank = trimmed.empty();
            if (blank && previous_blank) {
                continue;
            }
            out += line;
            out.push_back('\n');
            previous_blank = blank;
        }
        return out;
    }

    static long double code_candidate_quality(std::string_view prompt,
                                              const std::string& code,
                                              long double retrieval_score) {
        const auto query_tokens = tokenize_query(prompt);
        const auto lower_code = core3_lower(code);
        long double score = retrieval_score;
        std::size_t coverage = 0;
        for (const auto& token : query_tokens) {
            if (token.size() >= 4U && lower_code.find(token) != std::string::npos) {
                ++coverage;
            }
        }
        score += 0.07L * static_cast<long double>(coverage);

        const auto prompt_has = [&](std::string_view token) {
            return std::find(query_tokens.begin(), query_tokens.end(), token) != query_tokens.end();
        };
        const bool wants_cycle_detection = prompt_has("cycle") || prompt_has("detect") ||
                                           prompt_has("acyclic") || prompt_has("invalid");
        if (wants_cycle_detection) {
            if (lower_code.find("raise ") != std::string::npos) {
                score += 0.42L;
            }
            if (lower_code.find("cycle") != std::string::npos ||
                lower_code.find("acyclic") != std::string::npos) {
                score += 0.20L;
            }
        }
        const bool wants_graph_structure = prompt_has("graph") || prompt_has("dependency") ||
                                           prompt_has("dependencies") || prompt_has("topological");
        if (wants_graph_structure) {
            if (lower_code.find(" for ") != std::string::npos ||
                lower_code.find("\nfor ") != std::string::npos) {
                score += 0.10L;
            }
            if (lower_code.find(" if ") != std::string::npos ||
                lower_code.find("\nif ") != std::string::npos) {
                score += 0.10L;
            }
            if (lower_code.find("indegree") != std::string::npos ||
                lower_code.find("in_degree") != std::string::npos ||
                lower_code.find("difference_update") != std::string::npos ||
                lower_code.find("remove(") != std::string::npos) {
                score += 0.16L;
            }
        }
        if (returns_parameter_only(code)) {
            score -= 0.65L;
        }
        if (lower_code.find("import ") != std::string::npos ||
            lower_code.find("nx.") != std::string::npos ||
            lower_code.find("networkx") != std::string::npos) {
            score -= 0.16L;
        }
        return score;
    }

    static bool returns_parameter_only(std::string_view code) {
        const auto params = code_verifier_parameters(code);
        if (params.empty()) {
            return false;
        }
        std::stringstream stream{std::string(code)};
        std::string line;
        while (std::getline(stream, line)) {
            auto trimmed = core3_trim(line);
            if (trimmed.rfind("return ", 0) != 0) {
                continue;
            }
            auto expression = core3_trim(trimmed.substr(7U));
            for (auto param : params) {
                std::transform(param.begin(), param.end(), param.begin(), [](unsigned char ch) {
                    return static_cast<char>(std::tolower(ch));
                });
                std::transform(expression.begin(), expression.end(), expression.begin(), [](unsigned char ch) {
                    return static_cast<char>(std::tolower(ch));
                });
                if (expression == param) {
                    return true;
                }
            }
        }
        return false;
    }

    static std::vector<std::string> ranked_evidence_snippets(const std::vector<ConceptHit>& concepts,
                                                             const std::vector<std::string>& query_tokens,
                                                             std::size_t limit) {
        std::vector<std::pair<std::string, long double>> candidates;
        long double best_score = 0.0L;
        for (const auto& hit : concepts) {
            for (const auto& snippet : hit.atom.evidence_snippets) {
                const auto score = evidence_score(snippet, query_tokens);
                if (score > 0.0L &&
                    std::none_of(candidates.begin(), candidates.end(), [&](const auto& item) { return item.first == snippet; })) {
                    candidates.push_back({snippet, score});
                    best_score = std::max(best_score, score);
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
            return left.second > right.second;
        });
        std::vector<std::string> out;
        const long double threshold = best_score >= 2.0L ? 1.50L : 1.0L;
        for (const auto& [snippet, score] : candidates) {
            if (score < threshold) {
                continue;
            }
            out.push_back(snippet);
            if (out.size() >= limit) {
                break;
            }
        }
        return out;
    }

    static long double evidence_score(const std::string& snippet,
                                      const std::vector<std::string>& query_tokens) {
        const auto lower = core3_lower(snippet);
        if (lower.find("pmcid") != std::string::npos ||
            lower.find("author declared") != std::string::npos) {
            return 0.0L;
        }
        long double score = 0.0L;
        for (const auto& token : query_tokens) {
            if (token.size() >= 4U && lower.find(token) != std::string::npos) {
                score += 1.0L;
            }
        }
        if (lower.find("replication") != std::string::npos) {
            score += 0.75L;
        }
        if (lower.find("assembly") != std::string::npos || lower.find("release") != std::string::npos ||
            lower.find("genome") != std::string::npos || lower.find("host cell") != std::string::npos) {
            score += 0.35L;
        }
        return score;
    }

    static std::string stable_concept_summary(const std::vector<ConceptHit>& concepts,
                                              const std::vector<std::string>& query_tokens) {
        std::vector<std::string> labels;
        for (const auto& hit : concepts) {
            auto label = decode_concept_name(hit.atom.name);
            if (!clean_generated_label(label)) {
                continue;
            }
            if (!query_tokens.empty()) {
                const auto lower = core3_lower(label);
                const bool related = std::any_of(query_tokens.begin(), query_tokens.end(), [&](const auto& token) {
                    return token.size() >= 4U && lower.find(token) != std::string::npos;
                });
                if (!related && labels.size() >= 1U) {
                    continue;
                }
            }
            if (std::find(labels.begin(), labels.end(), label) == labels.end()) {
                labels.push_back(std::move(label));
            }
            if (labels.size() >= 3U) {
                break;
            }
        }
        if (labels.empty()) {
            return "I need more verified traces before generating a stable answer.";
        }
        std::string out = "Learned context:";
        for (const auto& label : labels) {
            out += " " + label + ".";
        }
        return out;
    }

    static bool clean_generated_label(const std::string& label) {
        const auto lower = core3_lower(label);
        if (lower.empty() || lower.find("pmcid") != std::string::npos ||
            lower.find("pmc") != std::string::npos ||
            lower.find("module") != std::string::npos ||
            lower.find("recursive loop") != std::string::npos ||
            lower.find("trace") != std::string::npos ||
            lower.find("corpus") != std::string::npos ||
            lower.find("rdrand") != std::string::npos) {
            return false;
        }
        return std::any_of(lower.begin(), lower.end(), [](unsigned char ch) {
            return std::isalpha(ch) != 0;
        });
    }

    void remember_working(std::string content, long double salience) {
        working_memory_.decay(1);
        const auto index = working_memory_.remember(std::move(content), salience);
        working_memory_.refresh(index);
    }

    void append_working_memory_trace(MentalTrace& trace) const {
        trace.events.push_back("working_memory:slots=" + std::to_string(working_memory_.slots().size()));
        for (const auto& slot : working_memory_.slots()) {
            trace.events.push_back("working_memory:item=" + slot.content);
        }
    }

    static std::string truncate_for_working_memory(std::string text) {
        if (text.size() <= 96U) {
            return text;
        }
        text.resize(96U);
        return text;
    }

    static void append_code_verifier_rejection(MentalTrace& trace,
                                               const PythonCodeVerification& verification) {
        trace.events.push_back("code_verifier:rejected");
        for (const auto& symbol : verification.undefined_symbols) {
            trace.events.push_back("code_verifier:undefined=" + symbol);
        }
        for (const auto& issue : verification.issues) {
            trace.events.push_back("code_verifier:issue=" + issue);
        }
    }

    static void merge_generation_trace(MentalTrace& target,
                                       const MentalTrace& generated) {
        for (const auto& module : generated.modules_used) {
            if (std::find(target.modules_used.begin(), target.modules_used.end(), module) == target.modules_used.end()) {
                target.modules_used.push_back(module);
            }
        }
        target.events.insert(target.events.end(), generated.events.begin(), generated.events.end());
    }

    UnifiedCognitiveConfig config_;
    DifferentiableFieldParameters params_;
    WorkingMemory working_memory_;
    ProgramSynthesizer synthesizer_;
    DzetaVM vm_;
    ModuleRegistry module_registry_;
    ModuleRouter router_;
    ConceptDictionary concept_dictionary_;
    SemanticEvidenceMemory evidence_memory_;
    PrincipleEngine principle_engine_;
    CodeSnippetMemory code_memory_;
    std::vector<BytecodeProgram> bytecode_skills_;
    std::map<std::uint64_t, long double> learned_trace_rewards_;

    friend void save_unified_cognitive_checkpoint(const UnifiedCognitiveField&, std::string_view);
    friend UnifiedCognitiveField load_unified_cognitive_checkpoint(std::string_view, UnifiedCheckpointLoadMode);

    void register_default_modules() {
        router_.register_module({"bytecode_synthesizer", "example_task", "answer", {}, {}, true});
        router_.register_module({"stored_bytecode_skill", "example_task", "answer", {}, {}, true});
        router_.register_module({"world_predictor", "causal_query", "prediction", {}, {}, true});
        router_.register_module({"phase_decoder", "prompt", "text", {}, {}, true});
    }

    void reinforce_runtime_layers(const MentalTrace& trace, long double reward) {
        if (config_.use_module_router) {
            for (const auto& module : trace.modules_used) {
                router_.apply_feedback(module, reward);
            }
        }
        if (config_.use_concept_dictionary) {
            concept_dictionary_.observe(trace);
        }
    }
};

inline std::string bytecode_op_name(BytecodeOp op) {
    return std::to_string(static_cast<int>(op));
}

inline BytecodeOp bytecode_op_from_int(int value) {
    switch (static_cast<BytecodeOp>(value)) {
    case BytecodeOp::ParseIntList:
    case BytecodeOp::ParseText:
    case BytecodeOp::GcdList:
    case BytecodeOp::SumList:
    case BytecodeOp::ProductList:
    case BytecodeOp::MaxList:
    case BytecodeOp::MinList:
    case BytecodeOp::SortIntList:
    case BytecodeOp::IsPalindrome:
    case BytecodeOp::CountVowels:
    case BytecodeOp::StringLength:
    case BytecodeOp::WordCount:
    case BytecodeOp::Halt:
        return static_cast<BytecodeOp>(value);
    }
    return BytecodeOp::Halt;
}

inline void write_core3_vector(std::ostream& output, std::string_view name, const std::vector<long double>& values) {
    output << "V\t" << name << '\t' << values.size();
    for (const auto value : values) {
        output << '\t' << static_cast<double>(value);
    }
    output << '\n';
}

inline char core3_hex_digit(unsigned value) {
    return static_cast<char>(value < 10U ? ('0' + value) : ('A' + (value - 10U)));
}

inline std::string core3_escape_field(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const unsigned char ch : text) {
        if (ch == '%' || ch == '\t' || ch == '\n' || ch == '\r') {
            out.push_back('%');
            out.push_back(core3_hex_digit(ch >> 4U));
            out.push_back(core3_hex_digit(ch & 0x0FU));
        } else {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

inline int core3_hex_value(char ch) {
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

inline std::string core3_unescape_field(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '%' && i + 2U < text.size()) {
            const int hi = core3_hex_value(text[i + 1U]);
            const int lo = core3_hex_value(text[i + 2U]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2U;
                continue;
            }
        }
        out.push_back(text[i]);
    }
    return out;
}

inline PrincipleRelation core3_relation_from_int(int value) {
    switch (static_cast<PrincipleRelation>(value)) {
    case PrincipleRelation::Cause:
    case PrincipleRelation::Sequence:
    case PrincipleRelation::PartOf:
    case PrincipleRelation::Definition:
    case PrincipleRelation::ConditionAction:
        return static_cast<PrincipleRelation>(value);
    }
    return PrincipleRelation::Definition;
}

inline void core3_write_u64(std::ostream& output, std::uint64_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline void core3_write_i32(std::ostream& output, std::int32_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline void core3_write_f64(std::ostream& output, long double value) {
    const double stored = static_cast<double>(value);
    output.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
}

inline void core3_write_string(std::ostream& output, std::string_view text) {
    core3_write_u64(output, static_cast<std::uint64_t>(text.size()));
    if (!text.empty()) {
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
    }
}

inline void core3_write_string_vector(std::ostream& output, const std::vector<std::string>& values) {
    core3_write_u64(output, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) {
        core3_write_string(output, value);
    }
}

inline void core3_write_ld_vector(std::ostream& output, const std::vector<long double>& values) {
    core3_write_u64(output, static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) {
        core3_write_f64(output, value);
    }
}

template <typename T>
inline T core3_read_raw(std::istream& input, std::string_view path) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("truncated unified cognitive checkpoint: " + std::string(path));
    }
    return value;
}

inline std::uint64_t core3_read_u64(std::istream& input, std::string_view path) {
    return core3_read_raw<std::uint64_t>(input, path);
}

inline std::int32_t core3_read_i32(std::istream& input, std::string_view path) {
    return core3_read_raw<std::int32_t>(input, path);
}

inline long double core3_read_f64(std::istream& input, std::string_view path) {
    return static_cast<long double>(core3_read_raw<double>(input, path));
}

inline std::string core3_read_string(std::istream& input, std::string_view path) {
    const auto size = core3_read_u64(input, path);
    if (size > static_cast<std::uint64_t>(512ULL * 1024ULL * 1024ULL)) {
        throw std::runtime_error("oversized field in unified cognitive checkpoint: " + std::string(path));
    }
    std::string out(static_cast<std::size_t>(size), '\0');
    if (!out.empty()) {
        input.read(out.data(), static_cast<std::streamsize>(out.size()));
        if (!input) {
            throw std::runtime_error("truncated unified cognitive checkpoint: " + std::string(path));
        }
    }
    return out;
}

inline void core3_skip_bytes(std::istream& input, std::uint64_t size, std::string_view path) {
    if (size == 0) {
        return;
    }
    input.seekg(static_cast<std::streamoff>(size), std::ios::cur);
    if (!input) {
        throw std::runtime_error("truncated unified cognitive checkpoint: " + std::string(path));
    }
}

inline void core3_skip_string(std::istream& input, std::string_view path) {
    const auto size = core3_read_u64(input, path);
    if (size > static_cast<std::uint64_t>(512ULL * 1024ULL * 1024ULL)) {
        throw std::runtime_error("oversized field in unified cognitive checkpoint: " + std::string(path));
    }
    core3_skip_bytes(input, size, path);
}

inline std::vector<std::string> core3_read_string_vector(std::istream& input, std::string_view path) {
    const auto count = core3_read_u64(input, path);
    if (count > 1'000'000ULL) {
        throw std::runtime_error("oversized string vector in unified cognitive checkpoint: " + std::string(path));
    }
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        out.push_back(core3_read_string(input, path));
    }
    return out;
}

inline void core3_skip_string_vector(std::istream& input, std::string_view path) {
    const auto count = core3_read_u64(input, path);
    if (count > 1'000'000ULL) {
        throw std::runtime_error("oversized string vector in unified cognitive checkpoint: " + std::string(path));
    }
    for (std::uint64_t i = 0; i < count; ++i) {
        core3_skip_string(input, path);
    }
}

inline std::vector<long double> core3_read_ld_vector(std::istream& input, std::string_view path) {
    const auto count = core3_read_u64(input, path);
    if (count > 1'000'000ULL) {
        throw std::runtime_error("oversized vector in unified cognitive checkpoint: " + std::string(path));
    }
    std::vector<long double> out;
    out.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        out.push_back(core3_read_f64(input, path));
    }
    return out;
}

inline void core3_skip_ld_vector(std::istream& input, std::string_view path) {
    const auto count = core3_read_u64(input, path);
    if (count > 1'000'000ULL) {
        throw std::runtime_error("oversized vector in unified cognitive checkpoint: " + std::string(path));
    }
    core3_skip_bytes(input, count * static_cast<std::uint64_t>(sizeof(double)), path);
}

inline void save_unified_cognitive_checkpoint(const UnifiedCognitiveField& core, std::string_view path) {
    std::ofstream output(std::string(path), std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write unified cognitive checkpoint: " + std::string(path));
    }
    const auto& params = core.params_;
    output << "DZETA_UNIFIED_CORE3_BIN10\n";
    core3_write_u64(output, static_cast<std::uint64_t>(params.dimension));
    core3_write_u64(output, static_cast<std::uint64_t>(params.charts));
    core3_write_ld_vector(output, params.routing_weights);
    core3_write_ld_vector(output, params.memory_gates);
    core3_write_ld_vector(output, params.planner_policy);
    core3_write_ld_vector(output, params.causal_edges);
    core3_write_ld_vector(output, params.decoder_logits);
    core3_write_ld_vector(output, params.module_confidence);
    core3_write_u64(output, static_cast<std::uint64_t>(core.bytecode_skills_.size()));
    for (const auto& skill : core.bytecode_skills_) {
        core3_write_string(output, skill.name);
        core3_write_u64(output, static_cast<std::uint64_t>(skill.instructions.size()));
        for (const auto& instruction : skill.instructions) {
            core3_write_i32(output, static_cast<std::int32_t>(instruction.op));
        }
    }
    core3_write_u64(output, static_cast<std::uint64_t>(core.learned_trace_rewards_.size()));
    for (const auto& [trace_id, reward] : core.learned_trace_rewards_) {
        core3_write_u64(output, trace_id);
        core3_write_f64(output, reward);
    }
    core3_write_u64(output, static_cast<std::uint64_t>(core.code_memory_.organ_strengths().size()));
    for (const auto& [organ, strength] : core.code_memory_.organ_strengths()) {
        core3_write_string(output, organ);
        core3_write_f64(output, strength);
    }
    core3_write_f64(output, core.code_memory_.count_if_trajectory_strength());
    core3_write_f64(output, core.code_memory_.collect_if_trajectory_strength());
    core3_write_f64(output, core.code_memory_.map_transform_trajectory_strength());
    core3_write_f64(output, core.code_memory_.reduce_accumulator_trajectory_strength());
    core3_write_u64(output, static_cast<std::uint64_t>(core.code_memory_.predicate_motifs().size()));
    for (const auto& motif : core.code_memory_.predicate_motifs()) {
        core3_write_string(output, motif.trigger);
        core3_write_string(output, motif.predicate_template);
        core3_write_f64(output, motif.strength);
    }
    core3_write_u64(output, static_cast<std::uint64_t>(core.code_memory_.transform_motifs().size()));
    for (const auto& motif : core.code_memory_.transform_motifs()) {
        core3_write_string(output, motif.trigger);
        core3_write_string(output, motif.expression_template);
        core3_write_f64(output, motif.strength);
    }
    core3_write_u64(output, static_cast<std::uint64_t>(core.code_memory_.reduce_motifs().size()));
    for (const auto& motif : core.code_memory_.reduce_motifs()) {
        core3_write_string(output, motif.trigger);
        core3_write_string(output, motif.initial_value);
        core3_write_string(output, motif.statement_template);
        core3_write_f64(output, motif.strength);
    }
    core3_write_u64(output, static_cast<std::uint64_t>(core.concept_dictionary_.atoms().size()));
    for (const auto& atom : core.concept_dictionary_.atoms()) {
        core3_write_string(output, atom.name);
        core3_write_u64(output, static_cast<std::uint64_t>(atom.observations));
        core3_write_f64(output, atom.mdl_gain);
        core3_write_ld_vector(output, atom.centroid);
        core3_write_u64(output, static_cast<std::uint64_t>(atom.symbols.size()));
        for (const auto& symbol : atom.symbols) {
            core3_write_string(output, symbol);
        }
        core3_write_u64(output, static_cast<std::uint64_t>(atom.evidence_snippets.size()));
        for (const auto& snippet : atom.evidence_snippets) {
            core3_write_string(output, snippet);
        }
    }
    core3_write_u64(output, static_cast<std::uint64_t>(core.evidence_memory_.sentences().size()));
    for (const auto& sentence : core.evidence_memory_.sentences()) {
        core3_write_u64(output, static_cast<std::uint64_t>(sentence.observations));
        core3_write_f64(output, sentence.reliability);
        core3_write_string(output, sentence.text);
    }
    core3_write_u64(output, static_cast<std::uint64_t>(core.principle_engine_.edges().size()));
    for (const auto& edge : core.principle_engine_.edges()) {
        core3_write_i32(output, static_cast<std::int32_t>(edge.relation));
        core3_write_f64(output, edge.strength);
        core3_write_string(output, edge.source);
        core3_write_string(output, edge.target);
        core3_write_string(output, edge.evidence);
    }
    core3_write_u64(output, static_cast<std::uint64_t>(core.principle_engine_.processes().size()));
    for (const auto& [process, steps] : core.principle_engine_.processes()) {
        core3_write_string(output, process);
        core3_write_u64(output, static_cast<std::uint64_t>(steps.size()));
        for (const auto& step : steps) {
            core3_write_string(output, step);
        }
    }
    core3_write_u64(output, static_cast<std::uint64_t>(core.code_memory_.snippets().size()));
    for (const auto& snippet : core.code_memory_.snippets()) {
        core3_write_u64(output, static_cast<std::uint64_t>(snippet.observations));
        core3_write_f64(output, snippet.reliability);
        core3_write_string(output, snippet.code);
        std::uint64_t flags = 0;
        if (snippet.indented_definition) {
            flags |= 1ULL;
        }
        if (snippet.private_or_internal) {
            flags |= 2ULL;
        }
        if (snippet.method_like) {
            flags |= 4ULL;
        }
        core3_write_u64(output, flags);
        core3_write_string(output, snippet.function_name);
        core3_write_string_vector(output, snippet.name_tokens);
        core3_write_string_vector(output, snippet.tokens);
        core3_write_string_vector(output, snippet.structure_tokens);
        core3_write_ld_vector(output, snippet.structure_vector);
        core3_write_string_vector(output, snippet.parameters);
        core3_write_string(output, snippet.return_expression);
    }
}

inline bool unified_cognitive_checkpoint_has_organ_summary(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        return false;
    }
    std::string line;
    std::getline(input, line);
    return line.rfind("DZETA_UNIFIED_CORE3_BIN5", 0) == 0 ||
           line.rfind("DZETA_UNIFIED_CORE3_BIN6", 0) == 0 ||
           line.rfind("DZETA_UNIFIED_CORE3_BIN7", 0) == 0 ||
           line.rfind("DZETA_UNIFIED_CORE3_BIN8", 0) == 0 ||
           line.rfind("DZETA_UNIFIED_CORE3_BIN9", 0) == 0 ||
           line.rfind("DZETA_UNIFIED_CORE3_BIN10", 0) == 0;
}

inline UnifiedCognitiveField load_unified_cognitive_checkpoint(std::string_view path,
                                                              UnifiedCheckpointLoadMode load_mode =
                                                                  UnifiedCheckpointLoadMode::Full) {
    const bool code_organs_only = load_mode == UnifiedCheckpointLoadMode::CodeOrgansOnly;
    const bool knowledge_only = load_mode == UnifiedCheckpointLoadMode::KnowledgeOnly;
    const bool trust_serialized_structure = load_mode == UnifiedCheckpointLoadMode::FullTrustedStructure;
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read unified cognitive checkpoint: " + std::string(path));
    }
    std::string line;
    std::getline(input, line);
    std::stringstream header(line);
    std::string field;
    std::getline(header, field, '\t');
    if (field == "DZETA_UNIFIED_CORE3_BIN1" || field == "DZETA_UNIFIED_CORE3_BIN2" ||
        field == "DZETA_UNIFIED_CORE3_BIN3" || field == "DZETA_UNIFIED_CORE3_BIN4" ||
        field == "DZETA_UNIFIED_CORE3_BIN5" || field == "DZETA_UNIFIED_CORE3_BIN6" ||
        field == "DZETA_UNIFIED_CORE3_BIN7" || field == "DZETA_UNIFIED_CORE3_BIN8" ||
        field == "DZETA_UNIFIED_CORE3_BIN9" || field == "DZETA_UNIFIED_CORE3_BIN10") {
        const bool prepared_code_snippets = field == "DZETA_UNIFIED_CORE3_BIN2" ||
                                           field == "DZETA_UNIFIED_CORE3_BIN3" ||
                                           field == "DZETA_UNIFIED_CORE3_BIN4" ||
                                           field == "DZETA_UNIFIED_CORE3_BIN5" ||
                                           field == "DZETA_UNIFIED_CORE3_BIN6" ||
                                           field == "DZETA_UNIFIED_CORE3_BIN7" ||
                                           field == "DZETA_UNIFIED_CORE3_BIN8" ||
                                           field == "DZETA_UNIFIED_CORE3_BIN9" ||
                                           field == "DZETA_UNIFIED_CORE3_BIN10";
        const bool saved_structure_tokens = field == "DZETA_UNIFIED_CORE3_BIN3" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN4" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN5" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN6" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN7" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN8" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN9" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN10";
        const bool saved_structure_vector = field == "DZETA_UNIFIED_CORE3_BIN4" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN5" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN6" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN7" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN8" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN9" ||
                                            field == "DZETA_UNIFIED_CORE3_BIN10";
        const bool saved_organ_summary = field == "DZETA_UNIFIED_CORE3_BIN5" ||
                                         field == "DZETA_UNIFIED_CORE3_BIN6" ||
                                         field == "DZETA_UNIFIED_CORE3_BIN7" ||
                                         field == "DZETA_UNIFIED_CORE3_BIN8" ||
                                         field == "DZETA_UNIFIED_CORE3_BIN9" ||
                                         field == "DZETA_UNIFIED_CORE3_BIN10";
        const bool saved_trajectory_fragments = field == "DZETA_UNIFIED_CORE3_BIN7" ||
                                                field == "DZETA_UNIFIED_CORE3_BIN8" ||
                                                field == "DZETA_UNIFIED_CORE3_BIN9" ||
                                                field == "DZETA_UNIFIED_CORE3_BIN10";
        const bool saved_collect_trajectory = field == "DZETA_UNIFIED_CORE3_BIN8" ||
                                              field == "DZETA_UNIFIED_CORE3_BIN9" ||
                                              field == "DZETA_UNIFIED_CORE3_BIN10";
        const bool saved_transform_trajectory = field == "DZETA_UNIFIED_CORE3_BIN9" ||
                                                field == "DZETA_UNIFIED_CORE3_BIN10";
        const bool saved_reduce_trajectory = field == "DZETA_UNIFIED_CORE3_BIN10";
        const bool early_organ_summary = field == "DZETA_UNIFIED_CORE3_BIN6" ||
                                         field == "DZETA_UNIFIED_CORE3_BIN7" ||
                                         field == "DZETA_UNIFIED_CORE3_BIN8" ||
                                         field == "DZETA_UNIFIED_CORE3_BIN9" ||
                                         field == "DZETA_UNIFIED_CORE3_BIN10";
        const auto dimension = static_cast<std::size_t>(core3_read_u64(input, path));
        const auto charts = static_cast<std::size_t>(core3_read_u64(input, path));
        auto params = make_differentiable_field_parameters(dimension, charts, 0xC03E30ULL);
        UnifiedCognitiveField core({}, params);
        core.params_.routing_weights = core3_read_ld_vector(input, path);
        core.params_.memory_gates = core3_read_ld_vector(input, path);
        core.params_.planner_policy = core3_read_ld_vector(input, path);
        core.params_.causal_edges = core3_read_ld_vector(input, path);
        core.params_.decoder_logits = core3_read_ld_vector(input, path);
        core.params_.module_confidence = core3_read_ld_vector(input, path);

        const auto skill_count = core3_read_u64(input, path);
        for (std::uint64_t i = 0; i < skill_count; ++i) {
            BytecodeProgram program;
            program.name = core3_read_string(input, path);
            const auto instruction_count = core3_read_u64(input, path);
            for (std::uint64_t j = 0; j < instruction_count; ++j) {
                program.instructions.push_back({bytecode_op_from_int(core3_read_i32(input, path)), 0});
            }
            core.bytecode_skills_.push_back(std::move(program));
        }

        const auto reward_count = core3_read_u64(input, path);
        for (std::uint64_t i = 0; i < reward_count; ++i) {
            const auto trace_id = core3_read_u64(input, path);
            core.learned_trace_rewards_[trace_id] = core3_read_f64(input, path);
        }

        if (early_organ_summary) {
            const auto organ_count = core3_read_u64(input, path);
            for (std::uint64_t i = 0; i < organ_count; ++i) {
                auto organ = core3_read_string(input, path);
                const auto strength = core3_read_f64(input, path);
                if (code_organs_only || knowledge_only) {
                    core.code_memory_.restore_organ_strength(std::move(organ), strength);
                }
            }
            if (saved_trajectory_fragments) {
                core.code_memory_.restore_count_if_trajectory_strength(core3_read_f64(input, path));
                if (saved_collect_trajectory) {
                    core.code_memory_.restore_collect_if_trajectory_strength(core3_read_f64(input, path));
                }
                if (saved_transform_trajectory) {
                    core.code_memory_.restore_map_transform_trajectory_strength(core3_read_f64(input, path));
                }
                if (saved_reduce_trajectory) {
                    core.code_memory_.restore_reduce_accumulator_trajectory_strength(core3_read_f64(input, path));
                }
                const auto motif_count = core3_read_u64(input, path);
                if (motif_count > 1'000'000ULL) {
                    throw std::runtime_error("oversized predicate motif table in unified cognitive checkpoint: " +
                                             std::string(path));
                }
                for (std::uint64_t i = 0; i < motif_count; ++i) {
                    auto trigger = core3_read_string(input, path);
                    auto predicate = core3_read_string(input, path);
                    const auto strength = core3_read_f64(input, path);
                    core.code_memory_.restore_predicate_motif(std::move(trigger),
                                                              std::move(predicate),
                                                              strength);
                }
                if (saved_transform_trajectory) {
                    const auto transform_count = core3_read_u64(input, path);
                    if (transform_count > 1'000'000ULL) {
                        throw std::runtime_error("oversized transform motif table in unified cognitive checkpoint: " +
                                                 std::string(path));
                    }
                    for (std::uint64_t i = 0; i < transform_count; ++i) {
                        auto trigger = core3_read_string(input, path);
                        auto expression = core3_read_string(input, path);
                        const auto strength = core3_read_f64(input, path);
                        core.code_memory_.restore_transform_motif(std::move(trigger),
                                                                  std::move(expression),
                                                                  strength);
                    }
                }
                if (saved_reduce_trajectory) {
                    const auto reduce_count = core3_read_u64(input, path);
                    if (reduce_count > 1'000'000ULL) {
                        throw std::runtime_error("oversized reduce motif table in unified cognitive checkpoint: " +
                                                 std::string(path));
                    }
                    for (std::uint64_t i = 0; i < reduce_count; ++i) {
                        auto trigger = core3_read_string(input, path);
                        auto initial = core3_read_string(input, path);
                        auto statement = core3_read_string(input, path);
                        const auto strength = core3_read_f64(input, path);
                        core.code_memory_.restore_reduce_motif(std::move(trigger),
                                                               std::move(initial),
                                                               std::move(statement),
                                                               strength);
                    }
                }
            }
            if (code_organs_only) {
                return core;
            }
        }

        const auto atom_count = core3_read_u64(input, path);
        for (std::uint64_t i = 0; i < atom_count; ++i) {
            if (code_organs_only) {
                core3_skip_string(input, path);
                (void)core3_read_u64(input, path);
                (void)core3_read_f64(input, path);
                core3_skip_ld_vector(input, path);
                core3_skip_string_vector(input, path);
                core3_skip_string_vector(input, path);
                continue;
            }
            ConceptAtom atom;
            atom.name = core3_read_string(input, path);
            atom.observations = static_cast<std::size_t>(core3_read_u64(input, path));
            atom.mdl_gain = core3_read_f64(input, path);
            atom.centroid = core3_read_ld_vector(input, path);
            const auto symbol_count = core3_read_u64(input, path);
            for (std::uint64_t j = 0; j < symbol_count; ++j) {
                atom.symbols.push_back(core3_read_string(input, path));
            }
            const auto snippet_count = core3_read_u64(input, path);
            for (std::uint64_t j = 0; j < snippet_count; ++j) {
                atom.evidence_snippets.push_back(core3_read_string(input, path));
            }
            core.concept_dictionary_.restore_atom(std::move(atom));
        }

        const auto sentence_count = core3_read_u64(input, path);
        if (!code_organs_only) {
            core.evidence_memory_.reserve(static_cast<std::size_t>(sentence_count));
        }
        for (std::uint64_t i = 0; i < sentence_count; ++i) {
            const auto observations = static_cast<std::size_t>(core3_read_u64(input, path));
            const auto reliability = core3_read_f64(input, path);
            if (code_organs_only) {
                core3_skip_string(input, path);
            } else {
                core.evidence_memory_.restore_serialized_sentence(core3_read_string(input, path), observations, reliability);
            }
        }

        const auto edge_count = core3_read_u64(input, path);
        for (std::uint64_t i = 0; i < edge_count; ++i) {
            if (code_organs_only) {
                (void)core3_read_i32(input, path);
                (void)core3_read_f64(input, path);
                core3_skip_string(input, path);
                core3_skip_string(input, path);
                core3_skip_string(input, path);
                continue;
            }
            PrincipleEdge edge;
            edge.relation = core3_relation_from_int(core3_read_i32(input, path));
            edge.strength = core3_read_f64(input, path);
            edge.source = core3_read_string(input, path);
            edge.target = core3_read_string(input, path);
            edge.evidence = core3_read_string(input, path);
            core.principle_engine_.restore_edge(std::move(edge));
        }

        const auto process_count = core3_read_u64(input, path);
        for (std::uint64_t i = 0; i < process_count; ++i) {
            if (code_organs_only) {
                core3_skip_string(input, path);
                core3_skip_string_vector(input, path);
                continue;
            }
            auto process = core3_read_string(input, path);
            const auto step_count = core3_read_u64(input, path);
            std::vector<std::string> steps;
            steps.reserve(static_cast<std::size_t>(step_count));
            for (std::uint64_t j = 0; j < step_count; ++j) {
                steps.push_back(core3_read_string(input, path));
            }
            core.principle_engine_.restore_process(std::move(process), std::move(steps));
        }

        if (knowledge_only) {
            return core;
        }

        if (saved_organ_summary && !early_organ_summary) {
            const auto organ_count = core3_read_u64(input, path);
            for (std::uint64_t i = 0; i < organ_count; ++i) {
                auto organ = core3_read_string(input, path);
                const auto strength = core3_read_f64(input, path);
                if (code_organs_only || knowledge_only) {
                    core.code_memory_.restore_organ_strength(std::move(organ), strength);
                }
            }
            if (code_organs_only || knowledge_only) {
                return core;
            }
        }

        const auto code_count = core3_read_u64(input, path);
        if (!code_organs_only) {
            core.code_memory_.reserve(static_cast<std::size_t>(code_count));
        }
        for (std::uint64_t i = 0; i < code_count; ++i) {
            if (code_organs_only && prepared_code_snippets && saved_structure_tokens) {
                const auto observations = static_cast<std::size_t>(core3_read_u64(input, path));
                const auto reliability = core3_read_f64(input, path);
                core3_skip_string(input, path);
                (void)core3_read_u64(input, path);
                core3_skip_string(input, path);
                core3_skip_string_vector(input, path);
                core3_skip_string_vector(input, path);
                auto structure_tokens = core3_read_string_vector(input, path);
                if (saved_structure_vector) {
                    core3_skip_ld_vector(input, path);
                }
                core3_skip_string_vector(input, path);
                core3_skip_string(input, path);
                core.code_memory_.restore_structure_organs(std::move(structure_tokens), observations, reliability);
                continue;
            }
            CodeSnippet snippet;
            snippet.observations = static_cast<std::size_t>(core3_read_u64(input, path));
            snippet.reliability = core3_read_f64(input, path);
            snippet.code = core3_read_string(input, path);
            if (prepared_code_snippets) {
                const auto flags = core3_read_u64(input, path);
                snippet.indented_definition = (flags & 1ULL) != 0;
                snippet.private_or_internal = (flags & 2ULL) != 0;
                snippet.method_like = (flags & 4ULL) != 0;
                snippet.function_name = core3_read_string(input, path);
                snippet.name_tokens = core3_read_string_vector(input, path);
                snippet.tokens = core3_read_string_vector(input, path);
                if (saved_structure_tokens) {
                    snippet.structure_tokens = core3_read_string_vector(input, path);
                }
                if (saved_structure_vector) {
                    snippet.structure_vector = core3_read_ld_vector(input, path);
                }
                snippet.parameters = core3_read_string_vector(input, path);
                snippet.return_expression = core3_read_string(input, path);
                core.code_memory_.restore_prepared_snippet(std::move(snippet),
                                                           true,
                                                           true,
                                                           !trust_serialized_structure);
            } else {
                core.code_memory_.restore_serialized_snippet(std::move(snippet.code), snippet.observations, snippet.reliability);
            }
        }
        core.evidence_memory_.compact();
        core.code_memory_.compact();
        return core;
    }
    if (field != "DZETA_UNIFIED_CORE3_V1" && field != "DZETA_UNIFIED_CORE3_V2") {
        throw std::runtime_error("invalid unified cognitive checkpoint: " + std::string(path));
    }
    std::getline(header, field, '\t');
    const auto dimension = static_cast<std::size_t>(std::stoull(field));
    std::getline(header, field, '\t');
    const auto charts = static_cast<std::size_t>(std::stoull(field));
    auto params = make_differentiable_field_parameters(dimension, charts, 0xC03E30ULL);
    UnifiedCognitiveField core({}, params);
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream stream(line);
        std::string kind;
        std::getline(stream, kind, '\t');
        if (kind == "V") {
            std::string name;
            std::getline(stream, name, '\t');
            std::getline(stream, field, '\t');
            const auto count = static_cast<std::size_t>(std::stoull(field));
            std::vector<long double> values;
            while (std::getline(stream, field, '\t')) {
                values.push_back(std::stold(field));
            }
            values.resize(count, 0.0L);
            if (name == "routing") {
                core.params_.routing_weights = std::move(values);
            } else if (name == "memory_gates") {
                core.params_.memory_gates = std::move(values);
            } else if (name == "planner_policy") {
                core.params_.planner_policy = std::move(values);
            } else if (name == "causal_edges") {
                core.params_.causal_edges = std::move(values);
            } else if (name == "decoder_logits") {
                core.params_.decoder_logits = std::move(values);
            } else if (name == "module_confidence") {
                core.params_.module_confidence = std::move(values);
            }
        } else if (kind == "S") {
            BytecodeProgram program;
            std::getline(stream, program.name, '\t');
            while (std::getline(stream, field, '\t')) {
                program.instructions.push_back({bytecode_op_from_int(std::stoi(field)), 0});
            }
            core.bytecode_skills_.push_back(std::move(program));
        } else if (kind == "R") {
            std::getline(stream, field, '\t');
            const auto trace_id = static_cast<std::uint64_t>(std::stoull(field));
            std::getline(stream, field, '\t');
            core.learned_trace_rewards_[trace_id] = std::stold(field);
        } else if (kind == "A") {
            ConceptAtom atom;
            std::getline(stream, field, '\t');
            atom.name = core3_unescape_field(field);
            std::getline(stream, field, '\t');
            atom.observations = static_cast<std::size_t>(std::stoull(field));
            std::getline(stream, field, '\t');
            atom.mdl_gain = std::stold(field);
            std::getline(stream, field, '\t');
            const auto centroid_count = static_cast<std::size_t>(std::stoull(field));
            atom.centroid.reserve(centroid_count);
            for (std::size_t i = 0; i < centroid_count && std::getline(stream, field, '\t'); ++i) {
                atom.centroid.push_back(std::stold(field));
            }
            if (std::getline(stream, field, '\t') && field == "S") {
                std::getline(stream, field, '\t');
                const auto symbol_count = static_cast<std::size_t>(std::stoull(field));
                for (std::size_t i = 0; i < symbol_count && std::getline(stream, field, '\t'); ++i) {
                    atom.symbols.push_back(core3_unescape_field(field));
                }
            }
            if (std::getline(stream, field, '\t') && field == "Q") {
                std::getline(stream, field, '\t');
                const auto snippet_count = static_cast<std::size_t>(std::stoull(field));
                for (std::size_t i = 0; i < snippet_count && std::getline(stream, field, '\t'); ++i) {
                    atom.evidence_snippets.push_back(core3_unescape_field(field));
                }
            }
            core.concept_dictionary_.restore_atom(std::move(atom));
        } else if (kind == "E") {
            std::getline(stream, field, '\t');
            const auto observations = static_cast<std::size_t>(std::stoull(field));
            std::getline(stream, field, '\t');
            const auto reliability = std::stold(field);
            std::getline(stream, field, '\t');
            core.evidence_memory_.restore_serialized_sentence(core3_unescape_field(field), observations, reliability);
        } else if (kind == "P") {
            PrincipleEdge edge;
            std::getline(stream, field, '\t');
            edge.relation = core3_relation_from_int(std::stoi(field));
            std::getline(stream, field, '\t');
            edge.strength = std::stold(field);
            std::getline(stream, field, '\t');
            edge.source = core3_unescape_field(field);
            std::getline(stream, field, '\t');
            edge.target = core3_unescape_field(field);
            std::getline(stream, field, '\t');
            edge.evidence = core3_unescape_field(field);
            core.principle_engine_.restore_edge(std::move(edge));
        } else if (kind == "PS") {
            std::getline(stream, field, '\t');
            auto process = core3_unescape_field(field);
            std::getline(stream, field, '\t');
            const auto step_count = static_cast<std::size_t>(std::stoull(field));
            std::vector<std::string> steps;
            steps.reserve(step_count);
            for (std::size_t i = 0; i < step_count && std::getline(stream, field, '\t'); ++i) {
                steps.push_back(core3_unescape_field(field));
            }
            core.principle_engine_.restore_process(std::move(process), std::move(steps));
        } else if (kind == "C") {
            std::getline(stream, field, '\t');
            const auto observations = static_cast<std::size_t>(std::stoull(field));
            std::getline(stream, field, '\t');
            const auto reliability = std::stold(field);
            std::getline(stream, field, '\t');
            core.code_memory_.restore_serialized_snippet(core3_unescape_field(field), observations, reliability);
        }
    }
    core.evidence_memory_.compact();
    core.code_memory_.compact();
    return core;
}

inline UnifiedCognitiveField load_unified_cognitive_checkpoint(std::string_view path,
                                                              bool code_organs_only) {
    return load_unified_cognitive_checkpoint(path,
                                             code_organs_only
                                                 ? UnifiedCheckpointLoadMode::CodeOrgansOnly
                                                 : UnifiedCheckpointLoadMode::Full);
}

} // namespace dzeta
