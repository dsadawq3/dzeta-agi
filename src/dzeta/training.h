#pragma once

#include "code_memory.h"
#include "field_objectives.h"
#include "iutt.h"
#include "math_learning.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct CorpusFile {
    std::string path;
    std::uintmax_t bytes = 0;
};

struct CorpusManifest {
    std::vector<CorpusFile> files;
    std::uintmax_t total_bytes = 0;
};

struct TrainingConfig {
    std::uintmax_t max_bytes = 200ULL * 1024ULL * 1024ULL;
    std::size_t chunk_bytes = 8192;
    std::size_t epochs = 3;
    std::size_t progress_every_chunks = 1000;
    std::vector<std::string> extensions{".py", ".pyi", ".pyx"};
};

struct TrainingReport {
    std::size_t epochs_completed = 0;
    std::size_t files_seen = 0;
    std::size_t chunks_seen = 0;
    std::uintmax_t bytes_seen = 0;
    std::size_t accepted_invariants = 0;
    std::size_t controlled_deaths = 0;
    long double mean_bridge_score = 0.0L;
};

struct DifferentiableCorpusTrainingReport {
    TrainingReport resonance;
    DifferentiableTrainingReport objective;
    long double initial_objective_loss = 0.0L;
    long double final_objective_loss = 0.0L;
};

inline std::string normalize_extension(std::string extension) {
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (!extension.empty() && extension.front() != '.') {
        extension.insert(extension.begin(), '.');
    }
    return extension;
}

inline bool is_corpus_file(const std::filesystem::path& path, const std::vector<std::string>& extensions) {
    const auto extension = normalize_extension(path.extension().string());
    for (auto allowed : extensions) {
        if (extension == normalize_extension(std::move(allowed))) {
            return true;
        }
    }
    return false;
}

inline bool is_code_corpus_file(const std::filesystem::path& path) {
    static const std::vector<std::string> code_extensions{".py", ".pyi", ".pyx"};
    return is_corpus_file(path, code_extensions);
}

inline CorpusManifest discover_corpus(std::string_view root_path,
                                      std::uintmax_t max_bytes,
                                      const std::vector<std::string>& extensions) {
    namespace fs = std::filesystem;
    const fs::path root{std::string(root_path)};
    if (!fs::exists(root)) {
        throw std::runtime_error("corpus path does not exist: " + root.string());
    }

    std::vector<CorpusFile> all_files;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file() || !is_corpus_file(entry.path(), extensions)) {
            continue;
        }
        all_files.push_back(CorpusFile{entry.path().string(), entry.file_size()});
    }

    std::sort(all_files.begin(), all_files.end(), [](const CorpusFile& left, const CorpusFile& right) {
        if (left.bytes == right.bytes) {
            return left.path < right.path;
        }
        return left.bytes > right.bytes;
    });

    CorpusManifest manifest;
    for (const auto& file : all_files) {
        if (max_bytes != 0 && manifest.total_bytes >= max_bytes) {
            break;
        }
        manifest.files.push_back(file);
        manifest.total_bytes += file.bytes;
    }
    return manifest;
}

inline CorpusManifest discover_corpus(std::string_view root_path, std::uintmax_t max_bytes) {
    static const std::vector<std::string> code_extensions{".py", ".pyi", ".pyx"};
    return discover_corpus(root_path, max_bytes, code_extensions);
}

inline std::string make_training_impulse(const CorpusFile& file,
                                         std::string_view chunk,
                                         std::size_t epoch,
                                         std::size_t chunk_index) {
    std::ostringstream impulse;
    impulse << "dzeta-corpus epoch=" << epoch
            << " chunk=" << chunk_index
            << " file=" << file.path
            << "\n";
    impulse << chunk;
    return impulse.str();
}

inline TrainingReport train_on_manifest(IuttEnsemble& ensemble,
                                        const CorpusManifest& manifest,
                                        const TrainingConfig& config,
                                        CodeTokenMemory* memory = nullptr) {
    if (config.chunk_bytes == 0) {
        throw std::runtime_error("training chunk_bytes must be greater than zero");
    }

    TrainingReport report;
    report.files_seen = manifest.files.size();
    long double score_sum = 0.0L;
    std::vector<char> buffer(config.chunk_bytes);

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::uintmax_t epoch_bytes = 0;
        for (const auto& file : manifest.files) {
            if (config.max_bytes != 0 && epoch_bytes >= config.max_bytes) {
                break;
            }

            std::ifstream input(file.path, std::ios::binary);
            if (!input) {
                continue;
            }
            while (input) {
                if (config.max_bytes != 0 && epoch_bytes >= config.max_bytes) {
                    break;
                }
                const auto remaining = config.max_bytes == 0
                                           ? static_cast<std::uintmax_t>(buffer.size())
                                           : std::min<std::uintmax_t>(buffer.size(), config.max_bytes - epoch_bytes);
                input.read(buffer.data(), static_cast<std::streamsize>(remaining));
                const auto read_count = input.gcount();
                if (read_count <= 0) {
                    break;
                }

                const std::string_view chunk(buffer.data(), static_cast<std::size_t>(read_count));
                const auto impulse = make_training_impulse(file, chunk, epoch, report.chunks_seen);
                const long double tick_time = 0.003L * static_cast<long double>((report.chunks_seen % 997U) + 1U);
                const auto result = ensemble.resonate(impulse, tick_time);

                if (result.accepted) {
                    ++report.accepted_invariants;
                    if (memory != nullptr) {
                        memory->observe(chunk);
                    }
                } else {
                    ++report.controlled_deaths;
                }
                score_sum += result.invariant.score;
                ++report.chunks_seen;
                report.bytes_seen += static_cast<std::uintmax_t>(read_count);
                epoch_bytes += static_cast<std::uintmax_t>(read_count);
            }
        }
        ++report.epochs_completed;
    }

    if (report.chunks_seen != 0) {
        report.mean_bridge_score = score_sum / static_cast<long double>(report.chunks_seen);
    }
    return report;
}

inline TrainingReport train_on_corpus(IuttEnsemble& ensemble,
                                      std::string_view root_path,
                                      const TrainingConfig& config,
                                      CodeTokenMemory* memory = nullptr) {
    const auto manifest = discover_corpus(root_path, config.max_bytes, config.extensions);
    return train_on_manifest(ensemble, manifest, config, memory);
}

inline TrainingReport field_train_on_manifest(IuttEnsemble& ensemble,
                                              const CorpusManifest& manifest,
                                              const TrainingConfig& config,
                                              SemanticFieldMemory& field_memory,
                                              std::size_t field_context = 512,
                                              CodeTokenMemory* token_memory = nullptr) {
    if (config.chunk_bytes == 0) {
        throw std::runtime_error("field training chunk_bytes must be greater than zero");
    }

    TrainingReport report;
    report.files_seen = manifest.files.size();
    long double score_sum = 0.0L;
    std::vector<char> buffer(config.chunk_bytes);

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::uintmax_t epoch_bytes = 0;
        for (const auto& file : manifest.files) {
            if (config.max_bytes != 0 && epoch_bytes >= config.max_bytes) {
                break;
            }
            std::ifstream input(file.path, std::ios::binary);
            if (!input) {
                continue;
            }
            while (input) {
                if (config.max_bytes != 0 && epoch_bytes >= config.max_bytes) {
                    break;
                }
                const auto remaining = config.max_bytes == 0
                                           ? static_cast<std::uintmax_t>(buffer.size())
                                           : std::min<std::uintmax_t>(buffer.size(), config.max_bytes - epoch_bytes);
                input.read(buffer.data(), static_cast<std::streamsize>(remaining));
                const auto read_count = input.gcount();
                if (read_count <= 0) {
                    break;
                }

                const std::string_view chunk(buffer.data(), static_cast<std::size_t>(read_count));
                const auto impulse = make_training_impulse(file, chunk, epoch, report.chunks_seen);
                const long double tick_time = 0.003L * static_cast<long double>((report.chunks_seen % 997U) + 1U);
                const auto result = ensemble.resonate(impulse, tick_time);
                auto field = make_field_state(ensemble.main_cloud().active_handles(field_context), impulse, field_context);
                learn_field_example(field_memory, field, impulse, std::string(chunk), result.accepted);
                if (token_memory != nullptr && result.accepted) {
                    token_memory->observe(chunk);
                }
                if (result.accepted) {
                    ++report.accepted_invariants;
                } else {
                    ++report.controlled_deaths;
                }
                score_sum += result.invariant.score;
                ++report.chunks_seen;
                report.bytes_seen += static_cast<std::uintmax_t>(read_count);
                epoch_bytes += static_cast<std::uintmax_t>(read_count);
            }
        }
        ++report.epochs_completed;
    }

    if (report.chunks_seen != 0) {
        report.mean_bridge_score = score_sum / static_cast<long double>(report.chunks_seen);
    }
    return report;
}

inline DifferentiableCorpusTrainingReport differentiable_field_train_on_manifest(
    IuttEnsemble& ensemble,
    const CorpusManifest& manifest,
    const TrainingConfig& config,
    SemanticFieldMemory& field_memory,
    DifferentiableFieldParameters& params,
    AdamWFieldOptimizer& optimizer,
    std::string_view objective,
    std::size_t steps_per_chunk = 1,
    std::size_t field_context = 512,
    CodeTokenMemory* token_memory = nullptr) {
    if (config.chunk_bytes == 0) {
        throw std::runtime_error("differentiable field training chunk_bytes must be greater than zero");
    }
    steps_per_chunk = std::max<std::size_t>(1, steps_per_chunk);

    DifferentiableCorpusTrainingReport out;
    out.resonance.files_seen = manifest.files.size();
    long double score_sum = 0.0L;
    long double initial_loss_sum = 0.0L;
    long double final_loss_sum = 0.0L;
    std::vector<char> buffer(config.chunk_bytes);

    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::uintmax_t epoch_bytes = 0;
        for (const auto& file : manifest.files) {
            if (config.max_bytes != 0 && epoch_bytes >= config.max_bytes) {
                break;
            }
            std::ifstream input(file.path, std::ios::binary);
            if (!input) {
                continue;
            }
            while (input) {
                if (config.max_bytes != 0 && epoch_bytes >= config.max_bytes) {
                    break;
                }
                const auto remaining = config.max_bytes == 0
                                           ? static_cast<std::uintmax_t>(buffer.size())
                                           : std::min<std::uintmax_t>(buffer.size(), config.max_bytes - epoch_bytes);
                input.read(buffer.data(), static_cast<std::streamsize>(remaining));
                const auto read_count = input.gcount();
                if (read_count <= 0) {
                    break;
                }

                const std::string_view chunk(buffer.data(), static_cast<std::size_t>(read_count));
                const auto impulse = make_training_impulse(file, chunk, epoch, out.resonance.chunks_seen);
                const long double tick_time = 0.003L * static_cast<long double>((out.resonance.chunks_seen % 997U) + 1U);
                const auto result = ensemble.resonate(impulse, tick_time);
                auto field = make_field_state(ensemble.main_cloud().active_handles(field_context), impulse, field_context);
                learn_field_example(field_memory, field, impulse, std::string(chunk), result.accepted);

                DifferentiableTrainingExample example;
                example.prompt = impulse;
                example.output = std::string(chunk);
                example.success = result.accepted;
                example.weight = result.accepted ? 1.0L : 0.35L;
                if (objective == "code" && !is_code_corpus_file(std::filesystem::path(file.path))) {
                    example.weight *= 0.25L;
                } else if (objective == "dialogue" && is_code_corpus_file(std::filesystem::path(file.path))) {
                    example.weight *= 0.35L;
                }
                const auto objective_report = train_differentiable_field(params, optimizer, {example}, steps_per_chunk);
                out.objective.steps += objective_report.steps;
                initial_loss_sum += objective_report.initial_loss;
                final_loss_sum += objective_report.final_loss;

                if (token_memory != nullptr && result.accepted) {
                    token_memory->observe(chunk);
                }
                if (result.accepted) {
                    ++out.resonance.accepted_invariants;
                } else {
                    ++out.resonance.controlled_deaths;
                }
                score_sum += result.invariant.score;
                ++out.resonance.chunks_seen;
                out.resonance.bytes_seen += static_cast<std::uintmax_t>(read_count);
                epoch_bytes += static_cast<std::uintmax_t>(read_count);
            }
        }
        ++out.resonance.epochs_completed;
    }

    if (out.resonance.chunks_seen != 0) {
        out.resonance.mean_bridge_score = score_sum / static_cast<long double>(out.resonance.chunks_seen);
        out.initial_objective_loss = initial_loss_sum / static_cast<long double>(out.resonance.chunks_seen);
        out.final_objective_loss = final_loss_sum / static_cast<long double>(out.resonance.chunks_seen);
        out.objective.initial_loss = out.initial_objective_loss;
        out.objective.final_loss = out.final_objective_loss;
        out.objective.mean_loss = 0.5L * (out.initial_objective_loss + out.final_objective_loss);
    }
    return out;
}

} // namespace dzeta
