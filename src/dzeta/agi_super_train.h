#pragma once

#include "executive.h"
#include "field_state.h"
#include "iutt.h"
#include "morphogenesis.h"
#include "principle_engine.h"
#include "self_creation.h"
#include "prf.h"
#include "unified_cognitive_field.h"
#include "variational_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct SuperTrainMetrics {
    std::uint64_t total_chunks = 0;
    std::uint64_t total_tokens_embedded = 0;
    long double mean_coherence = 0.0L;
    long double mean_field_energy = 0.0L;
    std::size_t concept_count = 0;
    std::size_t basin_count = 0;
    std::size_t field_size = 0;
    std::vector<long double> coherence_history;
};

struct SuperTrainConfig {
    std::string corpus_path;
    std::uintmax_t max_bytes = 0;
    std::size_t workers = 1;
    std::size_t print_interval = 100;
    std::size_t morphogenesis_interval = 1000;
    std::size_t save_interval = 5000;
    std::size_t max_iterations = 100000;
    std::size_t attractor_steps = 16;
};

struct SuperTrainSnapshot {
    SuperTrainMetrics metrics;
    std::string checkpoint_tag;
    std::chrono::steady_clock::time_point timestamp;
};

inline std::string load_file_text(const std::string& path, std::uintmax_t max_bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open corpus: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto file_size = static_cast<std::uintmax_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    const auto read_size = max_bytes == 0 ? file_size : std::min(file_size, max_bytes);
    const auto actual_read = std::min<std::uintmax_t>(read_size, 256ULL * 1024ULL * 1024ULL);
    std::string content(static_cast<std::size_t>(actual_read), '\0');
    input.read(content.data(), static_cast<std::streamsize>(actual_read));
    return content;
}

inline std::vector<std::string_view> split_chunks(std::string_view text, std::size_t chunk_size) {
    std::vector<std::string_view> chunks;
    chunk_size = std::max<std::size_t>(512, chunk_size);
    std::size_t pos = 0;
    while (pos < text.size()) {
        const auto end = std::min(pos + chunk_size, text.size());
        chunks.push_back(text.substr(pos, end - pos));
        pos = end;
    }
    if (chunks.empty() && !text.empty()) {
        chunks.push_back(text);
    }
    return chunks;
}

inline int run_agi_super_train(int argc, char** argv) {
    SuperTrainConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--corpus-path" && i + 1 < argc) {
            config.corpus_path = argv[++i];
            continue;
        }
        if (arg == "--train-bytes" && i + 1 < argc) {
            const auto raw = std::string(argv[++i]);
            auto value = raw;
            if (!value.empty()) {
                const char suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(value.back())));
                if (suffix == 'k' || suffix == 'm' || suffix == 'g') {
                    value.pop_back();
                    std::uintmax_t mult = 1;
                    if (suffix == 'k') mult = 1024;
                    else if (suffix == 'm') mult = 1024 * 1024;
                    else mult = 1024ULL * 1024 * 1024;
                    config.max_bytes = static_cast<std::uintmax_t>(std::stoull(value)) * mult;
                } else {
                    config.max_bytes = static_cast<std::uintmax_t>(std::stoull(value));
                }
            }
            continue;
        }
        if (arg == "--workers" && i + 1 < argc) {
            config.workers = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--print-interval" && i + 1 < argc) {
            config.print_interval = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--morphogenesis-interval" && i + 1 < argc) {
            config.morphogenesis_interval = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--save-interval" && i + 1 < argc) {
            config.save_interval = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--max-iterations" && i + 1 < argc) {
            config.max_iterations = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--attractor-steps" && i + 1 < argc) {
            config.attractor_steps = static_cast<std::size_t>(std::stoull(argv[++i]));
            continue;
        }
    }

    if (config.corpus_path.empty()) {
        std::cerr << "agi_super_train: --corpus-path is required\n";
        return 1;
    }

    const auto corpus_text = load_file_text(config.corpus_path, config.max_bytes);
    if (corpus_text.empty()) {
        std::cerr << "agi_super_train: empty corpus\n";
        return 1;
    }

    const auto chunks = split_chunks(corpus_text, 8192);
    std::cout << "agi_super_train corpus=" << config.corpus_path
              << " total_bytes=" << corpus_text.size()
              << " chunks=" << chunks.size() << "\n";

    DzetaConfig dzeta_cfg;
    dzeta_cfg.handle_count = 65536;
    dzeta_cfg.active_window = 4096;
    dzeta_cfg.temperature = 0.08L;
    dzeta_cfg.math_core_enabled = true;

    UnifiedCognitiveConfig core3_config;
    core3_config.max_concepts = 5000;
    core3_config.working_memory_slots = 7;
    core3_config.module_search_budget = 32;
    core3_config.recursive_depth = 4;

    IuttEnsemble ensemble(dzeta_cfg);
    UnifiedCognitiveField core3(core3_config);
    OscillatorField token_field(32768);
    PrincipleEngine principle_engine;
    ExecutiveConfig exec_config;
    exec_config.max_thoughts = 8;
    exec_config.broca_tokens = 48;
    FieldState field = make_seed_field_state("super_train_init", 128);

    SuperTrainMetrics metrics;
    std::size_t chunk_index = 0;
    const std::size_t total_chunks = chunks.size();
    long double running_coherence_sum = 0.0L;
    std::size_t coherence_samples = 0;

    const auto start_time = std::chrono::steady_clock::now();
    const auto log_time_ms = [&]() -> double {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start_time)
            .count();
    };

    std::cout << "agi_super_train_start\n";

    for (std::size_t iteration = 0; iteration < config.max_iterations; ++iteration) {
        const auto iter_start = std::chrono::steady_clock::now();

        const auto& current_chunk = chunks[chunk_index % total_chunks];
        ++chunk_index;

        prf.embed(current_chunk);

        principle_engine.observe_text(current_chunk);

        {
            const auto handles = ensemble.main_cloud().active_handles(256);
            if (!handles.empty()) {
                field = make_field_state(handles, current_chunk, 256);
            }
        }

        const auto attractor = run_variational_attractor(field, config.attractor_steps);
        const long double field_energy = attractor.final_energy.total_free_energy;

        const auto generated = prf.generate(field, 6);
        const auto report = prf.report_self();

        running_coherence_sum += report.field_coherence;
        ++coherence_samples;

        metrics.total_chunks = chunk_index;
        metrics.total_tokens_embedded += prf.size();
        metrics.mean_coherence = running_coherence_sum / static_cast<long double>(coherence_samples);
        metrics.mean_field_energy = field_energy;
        metrics.field_size = field.size();
        metrics.concept_count = principle_engine.edge_count();
        metrics.coherence_history.push_back(report.field_coherence);
        if (metrics.coherence_history.size() > 10000) {
            metrics.coherence_history.erase(metrics.coherence_history.begin(),
                                             metrics.coherence_history.begin() + 5000);
        }

        const auto basins = spectral_decode(field, 8);
        metrics.basin_count = basins.size();

        if (iteration % config.print_interval == 0) {
            const auto elapsed = log_time_ms();
            std::cout << "super_train_iteration=" << iteration
                      << " elapsed_ms=" << static_cast<std::uint64_t>(elapsed)
                      << " chunks=" << chunk_index
                      << " tokens=" << prf.size()
                      << " field_size=" << field.size()
                      << " coherence=" << static_cast<double>(report.field_coherence)
                      << " mean_coherence=" << static_cast<double>(metrics.mean_coherence)
                      << " field_energy=" << static_cast<double>(field_energy)
                      << " concepts=" << principle_engine.edge_count()
                      << " basins=" << basins.size()
                      << " accepted=" << (report.accepted ? "yes" : "no")
                      << "\n";

            const auto iter_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - iter_start)
                                     .count();
            std::cout << "super_train_perf iteration=" << iteration
                      << " iteration_ms=" << static_cast<double>(iter_ms)
                      << " tokens_per_sec="
                      << (iter_ms > 0.0
                              ? static_cast<double>(prf.size()) / (iter_ms / 1000.0)
                              : 0.0)
                      << "\n";
        }

        if (iteration % config.morphogenesis_interval == 0 && iteration != 0) {
            SemanticFieldMemory dummy_memory;
            const auto morph = apply_morphogenesis(field, dummy_memory,
                                                    evaluate_variational_energy(field), true);
            if (morph.changed) {
                std::cout << "super_train_morphogenesis iteration=" << iteration
                          << " born=" << morph.born_handles
                          << " split=" << morph.split_clusters
                          << " new_charts=" << morph.new_charts
                          << " organs=" << morph.memory_organs
                          << " pruned=" << morph.pruned_dead
                          << " entropy_before=" << static_cast<double>(morph.structural_entropy_before)
                          << " entropy_after=" << static_cast<double>(morph.structural_entropy_after)
                          << "\n";
            }

            const auto self_report = report.field_coherence < 0.30L
                ? self_create_from_failure(field, dummy_memory,
                                           std::string(current_chunk),
                                           generated, attractor.final_energy)
                : SelfCreationReport{};
            if (self_report.changed_geometry) {
                std::cout << "super_train_self_creation iteration=" << iteration
                          << " expanded_charts=" << self_report.expanded_charts
                          << " concept_organs=" << self_report.concept_organs
                          << " new_concepts=" << self_report.new_concepts
                          << "\n";
            }
        }

        if (iteration % config.save_interval == 0 && iteration != 0) {
            const auto elapsed = log_time_ms();
            std::cout << "super_train_snapshot iteration=" << iteration
                      << " elapsed_ms=" << static_cast<std::uint64_t>(elapsed)
                      << " coherence=" << static_cast<double>(metrics.mean_coherence)
                      << " concepts=" << metrics.concept_count
                      << " field_size=" << metrics.field_size
                      << "\n";
        }

        if (iteration % 100 == 0 && iteration != 0) {
            const std::string assessment_queries[] = {
                "what is the main topic of the material",
                "summarize the key concepts",
                "what relationships exist between the ideas",
            };
            const auto& query = assessment_queries[chunk_index % 3];
            const auto principle_answer = principle_engine.answer(query);
            if (principle_answer) {
                const long double confidence = principle_answer->confidence;
                const long double word_ratio = principle_answer->text.empty()
                    ? 0.0L
                    : std::min(1.0L, static_cast<long double>(principle_answer->text.size()) / 200.0L);
                const long double self_score = 0.50L * confidence + 0.50L * word_ratio;
                std::cout << "super_train_self_assessment iteration=" << iteration
                          << " query=\"" << query << "\""
                          << " confidence=" << static_cast<double>(confidence)
                          << " response_length=" << principle_answer->text.size()
                          << " self_score=" << static_cast<double>(self_score)
                          << " coherence=" << static_cast<double>(report.field_coherence)
                          << "\n";
            }
        }
    }

    const auto total_elapsed = log_time_ms();
    std::cout << "agi_super_train_end"
              << " total_iterations=" << config.max_iterations
              << " total_chunks=" << chunk_index
              << " total_elapsed_ms=" << static_cast<std::uint64_t>(total_elapsed)
              << " final_coherence=" << static_cast<double>(metrics.mean_coherence)
              << " final_concepts=" << metrics.concept_count
              << " final_tokens=" << metrics.total_tokens_embedded
              << "\n";

    return 0;
}

} // namespace dzeta
