#pragma once

#include "morphogenesis.h"
#include "principle_engine.h"
#include "self_creation.h"
#include "semantic_evidence_memory.h"
#include "semantic_field_memory.h"
#include "token_field.h"
#include "variational_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dzeta {

struct LearningConfig {
    std::size_t max_tokens = 8192;
    std::size_t max_evidence_sentences = 4096;
    std::size_t attractor_steps = 20;
    std::size_t eval_interval = 10;
    std::size_t self_eval_interval = 5;
    std::size_t max_basins = 6;
    bool morphogenesis_enabled = true;
    bool self_creation_enabled = true;
    bool print_progress = true;
    long double coherence_threshold = 0.28L;
    long double attractor_lr = 0.08L;
};

struct LearningMetricsSnapshot {
    std::size_t iteration = 0;
    long double coherence = 0.0L;
    std::size_t tokens_activated = 0;
    long double field_energy = 0.0L;
    std::size_t principle_edges = 0;
    std::size_t evidence_sentences = 0;
    std::size_t memory_attractors = 0;
    long double self_assessment_coherence = 0.0L;
    std::size_t field_size = 0;
    long double free_energy = 0.0L;
};

class ContinuousLearner {
public:
    explicit ContinuousLearner(LearningConfig config = {})
        : config_(config),
          field_(config_.max_tokens),
          evidence_(config_.max_evidence_sentences) {}

    void learn_corpus(const std::vector<std::string>& corpus) {
        const std::size_t total = corpus.size();
        for (std::size_t i = 0; i < total; ++i) {
            const auto metrics = learn_chunk(corpus[i], i);

            if (config_.print_progress && (i % config_.eval_interval == 0 || i == total - 1)) {
                std::cout << "[continuous_learning] iter=" << metrics.iteration
                          << " coherence=" << static_cast<double>(metrics.coherence)
                          << " activated=" << metrics.tokens_activated
                          << " field_energy=" << static_cast<double>(metrics.field_energy)
                          << " free_energy=" << static_cast<double>(metrics.free_energy)
                          << " principles=" << metrics.principle_edges
                          << " evidence=" << metrics.evidence_sentences
                          << " attractors=" << metrics.memory_attractors
                          << "\n";
            }
        }
    }

    void learn_corpus_from_file(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            std::cerr << "[continuous_learning] cannot open corpus: " << path << "\n";
            return;
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty()) {
                lines.push_back(std::move(line));
            }
        }
        learn_corpus(lines);
    }

    LearningMetricsSnapshot learn_chunk(const std::string& text, std::size_t iteration) {
        field_.embed(text);
        principles_.observe_text(text);
        evidence_.observe_text(text);

        const auto field_report = field_.report_self();
        auto field_energy = evaluate_variational_energy(current_field_);

        if (!current_field_.empty()) {
            const auto attractor = run_variational_attractor(current_field_, config_.attractor_steps);
            field_energy = attractor.final_energy;
        }

        if (config_.morphogenesis_enabled) {
            apply_morphogenesis(current_field_, field_memory_, field_energy, true);
        }

        if (config_.self_creation_enabled && field_energy.stability > 0.45L) {
            const auto generated = field_.generate_self();
            if (!generated.empty()) {
                self_create_from_success(current_field_, field_memory_, text, generated);
            }
        }

        field_memory_.imprint(text, current_field_, field_energy,
                              field_.generate_self(), field_energy.stability > 0.4L);

        LearningMetricsSnapshot snap;
        snap.iteration = iteration;
        snap.coherence = field_report.field_coherence;
        snap.tokens_activated = field_report.tokens_activated;
        snap.field_energy = field_energy.total_free_energy;
        snap.free_energy = field_energy.total_free_energy;
        snap.principle_edges = principles_.edge_count();
        snap.evidence_sentences = evidence_.size();
        snap.memory_attractors = field_memory_.size();
        snap.field_size = current_field_.size();

        if (iteration > 0 && iteration % config_.self_eval_interval == 0) {
            snap.self_assessment_coherence = self_evaluate();
        }

        history_.push_back(snap);
        return snap;
    }

    long double self_evaluate() {
        const auto concepts = extract_concept_labels();
        if (concepts.empty()) {
            return 0.0L;
        }

        std::mt19937_64 rng(static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        std::uniform_int_distribution<std::size_t> dist(0, concepts.size() - 1);

        long double total_coherence = 0.0L;
        std::size_t evaluated = 0;

        const std::size_t max_questions = std::min<std::size_t>(3, concepts.size());
        std::unordered_set<std::size_t> asked;
        for (std::size_t q = 0; q < max_questions * 5 && asked.size() < max_questions; ++q) {
            const std::size_t idx = dist(rng);
            if (!asked.insert(idx).second) continue;

            const auto& concept = concepts[idx];
            const auto question = generate_self_question(concept);
            const auto answer = principles_.answer(question);

            if (answer) {
                const auto answer_tokens = tokenize_query(answer->text);
                const auto concept_tokens = tokenize_query(concept);
                long double overlap = 0.0L;
                if (!concept_tokens.empty() && !answer_tokens.empty()) {
                    std::size_t shared = 0;
                    for (const auto& tok : concept_tokens) {
                        if (std::find(answer_tokens.begin(), answer_tokens.end(), tok) != answer_tokens.end()) {
                            ++shared;
                        }
                    }
                    overlap = static_cast<long double>(shared) /
                              static_cast<long double>(concept_tokens.size());
                }
                total_coherence += 0.5L * overlap + 0.5L * answer->confidence;
                ++evaluated;
            } else {
                const auto generated = field_.generate_self();
                if (!generated.empty()) {
                    const auto generated_tokens = tokenize_query(generated);
                    const auto concept_tokens = tokenize_query(concept);
                    long double overlap = 0.0L;
                    if (!concept_tokens.empty() && !generated_tokens.empty()) {
                        std::size_t shared = 0;
                        for (const auto& tok : concept_tokens) {
                            if (std::find(generated_tokens.begin(), generated_tokens.end(), tok) != generated_tokens.end()) {
                                ++shared;
                            }
                        }
                        overlap = static_cast<long double>(shared) /
                                  static_cast<long double>(concept_tokens.size());
                    }
                    total_coherence += overlap * 0.6L;
                    ++evaluated;
                }
            }
        }

        return evaluated > 0 ? total_coherence / static_cast<long double>(evaluated) : 0.0L;
    }

    bool save_checkpoint(const std::string& path) {
        std::ofstream output(path, std::ios::binary);
        if (!output) {
            std::cerr << "[continuous_learning] cannot write checkpoint: " << path << "\n";
            return false;
        }

        output << "DZETA_CONTINUOUS_LEARNER_V1\n";
        output << "max_tokens=" << config_.max_tokens << "\n";
        output << "history_size=" << history_.size() << "\n";

        output << "principles_edges=" << principles_.edge_count() << "\n";
        for (const auto& edge : principles_.edges()) {
            output << "edge\t" << static_cast<int>(edge.relation)
                   << "\t" << semantic_memory_hex_encode(edge.source)
                   << "\t" << semantic_memory_hex_encode(edge.target)
                   << "\t" << static_cast<double>(edge.strength) << "\n";
        }

        output << "principle_processes=" << principles_.process_count() << "\n";
        for (const auto& [process, steps] : principles_.processes()) {
            output << "process\t" << semantic_memory_hex_encode(process);
            for (const auto& step : steps) {
                output << "\t" << semantic_memory_hex_encode(step);
            }
            output << "\n";
        }

        output << "history_entries=" << history_.size() << "\n";
        for (const auto& snap : history_) {
            output << "snap\t" << snap.iteration
                   << "\t" << static_cast<double>(snap.coherence)
                   << "\t" << snap.tokens_activated
                   << "\t" << static_cast<double>(snap.field_energy)
                   << "\t" << snap.principle_edges
                   << "\t" << snap.evidence_sentences
                   << "\t" << snap.memory_attractors
                   << "\t" << static_cast<double>(snap.self_assessment_coherence)
                   << "\t" << snap.field_size
                   << "\t" << static_cast<double>(snap.free_energy) << "\n";
        }

        output << "checkpoint_end\n";
        return true;
    }

    bool load_checkpoint(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            std::cerr << "[continuous_learning] cannot read checkpoint: " << path << "\n";
            return false;
        }

        std::string header;
        std::getline(input, header);
        if (header != "DZETA_CONTINUOUS_LEARNER_V1") {
            std::cerr << "[continuous_learning] invalid checkpoint header\n";
            return false;
        }

        history_.clear();

        std::string line;
        while (std::getline(input, line)) {
            if (line == "checkpoint_end") break;
            if (line.empty()) continue;

            if (line.rfind("max_tokens=", 0) == 0) {
                config_.max_tokens = static_cast<std::size_t>(std::stoull(line.substr(11)));
            } else if (line.rfind("principles_edges=", 0) == 0) {
                // edges follow on separate lines
            } else if (line.rfind("edge\t", 0) == 0) {
                std::vector<std::string> parts;
                std::size_t begin = 5;
                for (;;) {
                    const auto tab = line.find('\t', begin);
                    if (tab == std::string::npos) {
                        parts.push_back(line.substr(begin));
                        break;
                    }
                    parts.push_back(line.substr(begin, tab - begin));
                    begin = tab + 1;
                }
                if (parts.size() >= 4) {
                    PrincipleEdge edge;
                    edge.relation = static_cast<PrincipleRelation>(std::stoi(parts[0]));
                    edge.source = semantic_memory_hex_decode(parts[1]);
                    edge.target = semantic_memory_hex_decode(parts[2]);
                    edge.strength = std::stold(parts[3]);
                    principles_.restore_edge(edge);
                }
            } else if (line.rfind("process\t", 0) == 0) {
                const auto rest = line.substr(8);
                std::vector<std::string> parts;
                std::size_t begin = 0;
                for (;;) {
                    const auto tab = rest.find('\t', begin);
                    if (tab == std::string::npos) {
                        parts.push_back(rest.substr(begin));
                        break;
                    }
                    parts.push_back(rest.substr(begin, tab - begin));
                    begin = tab + 1;
                }
                if (parts.size() >= 2) {
                    const auto process = semantic_memory_hex_decode(parts[0]);
                    std::vector<std::string> steps;
                    for (std::size_t i = 1; i < parts.size(); ++i) {
                        steps.push_back(semantic_memory_hex_decode(parts[i]));
                    }
                    principles_.restore_process(process, steps);
                }
            } else if (line.rfind("snap\t", 0) == 0) {
                const auto rest = line.substr(5);
                std::vector<std::string> parts;
                std::size_t begin = 0;
                for (;;) {
                    const auto tab = rest.find('\t', begin);
                    if (tab == std::string::npos) {
                        parts.push_back(rest.substr(begin));
                        break;
                    }
                    parts.push_back(rest.substr(begin, tab - begin));
                    begin = tab + 1;
                }
                if (parts.size() >= 10) {
                    LearningMetricsSnapshot snap;
                    snap.iteration = static_cast<std::size_t>(std::stoull(parts[0]));
                    snap.coherence = std::stold(parts[1]);
                    snap.tokens_activated = static_cast<std::size_t>(std::stoull(parts[2]));
                    snap.field_energy = std::stold(parts[3]);
                    snap.principle_edges = static_cast<std::size_t>(std::stoull(parts[4]));
                    snap.evidence_sentences = static_cast<std::size_t>(std::stoull(parts[5]));
                    snap.memory_attractors = static_cast<std::size_t>(std::stoull(parts[6]));
                    snap.self_assessment_coherence = std::stold(parts[7]);
                    snap.field_size = static_cast<std::size_t>(std::stoull(parts[8]));
                    snap.free_energy = std::stold(parts[9]);
                    history_.push_back(std::move(snap));
                }
            }
        }

        return true;
    }

    const std::vector<LearningMetricsSnapshot>& history() const noexcept { return history_; }
    PhaseResonantField& token_field() noexcept { return field_; }
    const PhaseResonantField& token_field() const noexcept { return field_; }
    PrincipleEngine& principle_engine() noexcept { return principles_; }
    const PrincipleEngine& principle_engine() const noexcept { return principles_; }
    SemanticEvidenceMemory& evidence_memory() noexcept { return evidence_; }
    const SemanticEvidenceMemory& evidence_memory() const noexcept { return evidence_; }
    SemanticFieldMemory& field_memory() noexcept { return field_memory_; }
    const SemanticFieldMemory& field_memory() const noexcept { return field_memory_; }
    FieldState& field_state() noexcept { return current_field_; }
    const FieldState& field_state() const noexcept { return current_field_; }
    const LearningConfig& config() const noexcept { return config_; }

private:
    LearningConfig config_;
    PhaseResonantField field_;
    PrincipleEngine principles_;
    SemanticEvidenceMemory evidence_;
    SemanticFieldMemory field_memory_;
    FieldState current_field_;
    std::vector<LearningMetricsSnapshot> history_;

    std::vector<std::string> extract_concept_labels() const {
        std::vector<std::string> labels;
        if (current_field_.empty()) {
            for (const auto& th : field_.labels()) {
                if (th.amplitude > 0.25L && th.token.size() >= 3) {
                    labels.push_back(th.token);
                    if (labels.size() >= 16) break;
                }
            }
            return labels;
        }
        const auto concepts = infer_concepts_from_field(current_field_, field_memory_);
        for (const auto& c : concepts) {
            if (labels.size() >= 16) break;
            labels.push_back(c);
        }
        return labels;
    }

    static std::string generate_self_question(const std::string& concept) {
        static const std::vector<std::string> templates = {
            "What is ",
            "Define ",
            "Explain ",
            "Describe "
        };
        std::mt19937_64 rng(static_cast<std::uint64_t>(
            std::hash<std::string>{}(concept)));
        std::uniform_int_distribution<std::size_t> dist(0, templates.size() - 1);
        return templates[dist(rng)] + concept + ".";
    }
};

inline LearningMetricsSnapshot run_agi_learning_session(
    const std::vector<std::string>& corpus,
    std::size_t iterations,
    LearningConfig config) {

    ContinuousLearner learner(config);
    const std::size_t total = std::min(iterations, corpus.size());

    for (std::size_t i = 0; i < total; ++i) {
        learner.learn_chunk(corpus[i], i);

        if (config.print_progress && ((i + 1) % config.eval_interval == 0 || i == total - 1)) {
            const auto& hist = learner.history();
            const auto& last = hist.back();
            std::cout << "[agi_session] progress=" << (i + 1) << "/" << total
                      << " coherence=" << static_cast<double>(last.coherence)
                      << " activated=" << last.tokens_activated
                      << " free_energy=" << static_cast<double>(last.free_energy)
                      << " self_eval=" << static_cast<double>(last.self_assessment_coherence)
                      << " principles=" << last.principle_edges
                      << " evidence=" << last.evidence_sentences
                      << "\n";
        }

        if ((i + 1) % 10 == 0) {
            std::cout << "[agi_report] iteration=" << (i + 1)
                      << " coherence=" << static_cast<double>(learner.history().back().coherence)
                      << " tokens_activated=" << learner.history().back().tokens_activated
                      << " free_energy=" << static_cast<double>(learner.history().back().free_energy)
                      << "\n";
        }
    }

    if (!learner.history().empty()) {
        return learner.history().back();
    }
    return {};
}

inline LearningMetricsSnapshot run_agi_learning_session(
    const std::string& corpus_path,
    std::size_t iterations,
    LearningConfig config) {

    std::ifstream input(corpus_path, std::ios::binary);
    if (!input) {
        std::cerr << "[agi_session] cannot open corpus: " << corpus_path << "\n";
        return {};
    }

    std::vector<std::string> corpus;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            corpus.push_back(std::move(line));
        }
    }

    return run_agi_learning_session(corpus, iterations, config);
}

inline std::string generate_agi_report(const ContinuousLearner& learner) {
    std::ostringstream out;
    const auto& history = learner.history();

    out << "========================================\n";
    out << "   DZETA AGI CONTINUOUS LEARNING REPORT\n";
    out << "========================================\n\n";

    out << "Configuration:\n";
    out << "  Max tokens:      " << learner.config().max_tokens << "\n";
    out << "  Max evidence:    " << learner.config().max_evidence_sentences << "\n";
    out << "  Attractor steps: " << learner.config().attractor_steps << "\n";
    out << "  Eval interval:   " << learner.config().eval_interval << "\n";
    out << "  Morphogenesis:   " << (learner.config().morphogenesis_enabled ? "enabled" : "disabled") << "\n";
    out << "  Self-creation:   " << (learner.config().self_creation_enabled ? "enabled" : "disabled") << "\n\n";

    if (history.empty()) {
        out << "No learning history.\n";
        out << "========================================\n";
        return out.str();
    }

    const auto& first = history.front();
    const auto& last = history.back();

    out << "Learning Progress (" << history.size() << " iterations):\n\n";

    out << "  Initial state:\n";
    out << "    Coherence:         " << std::fixed << std::setprecision(4)
        << static_cast<double>(first.coherence) << "\n";
    out << "    Tokens activated:  " << first.tokens_activated << "\n";
    out << "    Free energy:       " << std::fixed << std::setprecision(4)
        << static_cast<double>(first.free_energy) << "\n";
    out << "    Principle edges:   " << first.principle_edges << "\n";
    out << "    Evidence:          " << first.evidence_sentences << "\n";
    out << "    Memory attractors: " << first.memory_attractors << "\n\n";

    out << "  Final state:\n";
    out << "    Coherence:         " << std::fixed << std::setprecision(4)
        << static_cast<double>(last.coherence) << "\n";
    out << "    Tokens activated:  " << last.tokens_activated << "\n";
    out << "    Free energy:       " << std::fixed << std::setprecision(4)
        << static_cast<double>(last.free_energy) << "\n";
    out << "    Principle edges:   " << last.principle_edges << "\n";
    out << "    Evidence:          " << last.evidence_sentences << "\n";
    out << "    Memory attractors: " << last.memory_attractors << "\n";
    out << "    Self-assessment:   " << std::fixed << std::setprecision(4)
        << static_cast<double>(last.self_assessment_coherence) << "\n";
    out << "    Field size:        " << last.field_size << "\n\n";

    long double avg_coherence = 0.0L;
    long double best_coherence = 0.0L;
    for (const auto& snap : history) {
        avg_coherence += snap.coherence;
        if (snap.coherence > best_coherence) {
            best_coherence = snap.coherence;
        }
    }
    avg_coherence /= static_cast<long double>(history.size());

    out << "Summary:\n";
    out << "  Average coherence:     " << std::fixed << std::setprecision(4)
        << static_cast<double>(avg_coherence) << "\n";
    out << "  Peak coherence:        " << std::fixed << std::setprecision(4)
        << static_cast<double>(best_coherence) << "\n";
    out << "  Total principle edges: " << learner.principle_engine().edge_count() << "\n";
    out << "  Total evidence:        " << learner.evidence_memory().size() << "\n";
    out << "  Total attractors:      " << learner.field_memory().size() << "\n";
    out << "  Field tokens:          " << learner.token_field().size() << "\n\n";

    out << "History:\n";
    out << "  iter  coherence  activated  free_energy  principles  evidence  attractors  self_eval\n";
    for (const auto& snap : history) {
        out << "  " << std::setw(4) << snap.iteration
            << "  " << std::fixed << std::setprecision(3) << std::setw(8)
            << static_cast<double>(snap.coherence)
            << "  " << std::setw(8) << snap.tokens_activated
            << "  " << std::setw(10) << static_cast<double>(snap.free_energy)
            << "  " << std::setw(9) << snap.principle_edges
            << "  " << std::setw(8) << snap.evidence_sentences
            << "  " << std::setw(9) << snap.memory_attractors
            << "  " << std::setw(9) << static_cast<double>(snap.self_assessment_coherence)
            << "\n";
    }

    out << "========================================\n";
    return out.str();
}

} // namespace dzeta
