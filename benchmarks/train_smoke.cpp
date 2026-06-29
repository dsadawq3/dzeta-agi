#include "token_field.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path corpus_path;
    std::size_t oscillators = 65536;
    std::size_t dimensions = 192;
    std::size_t seconds = 60;
    std::size_t max_lines = 0;
    std::size_t max_line_chars = 512;
    std::size_t progress_seconds = 30;
    std::size_t tokens = 12;
    std::uint64_t seed = 0;
    long double temperature = 0.08L;
    long double learning_rate = 0.32L;
    std::size_t threads = 0;
    std::size_t parallel_min_dimensions = 2048;
};

void print_usage() {
    std::cout
        << "Usage: dzeta_train_smoke --corpus PATH [--seconds N] [--oscillators N]\n"
        << "                         [--dimensions N] [--max-lines N]\n"
        << "                         [--max-line-chars N] [--progress-seconds N]\n"
        << "                         [--tokens N] [--seed N]\n"
        << "                         [--temperature X] [--learning-rate X]\n"
        << "                         [--threads N] [--parallel-min-dim N]\n";
}

std::size_t parse_size(std::string_view value) {
    if (value.empty()) {
        throw std::runtime_error("empty numeric value");
    }
    return static_cast<std::size_t>(std::stoull(std::string(value)));
}

long double parse_float(std::string_view value) {
    if (value.empty()) {
        throw std::runtime_error("empty floating-point value");
    }
    return std::stold(std::string(value));
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto require_value = [&](std::string_view name) -> char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        }
        if (arg == "--corpus") {
            options.corpus_path = require_value(arg);
        } else if (arg == "--seconds") {
            options.seconds = parse_size(require_value(arg));
        } else if (arg == "--oscillators") {
            options.oscillators = parse_size(require_value(arg));
        } else if (arg == "--dimensions") {
            options.dimensions = parse_size(require_value(arg));
        } else if (arg == "--max-lines") {
            options.max_lines = parse_size(require_value(arg));
        } else if (arg == "--max-line-chars") {
            options.max_line_chars = parse_size(require_value(arg));
        } else if (arg == "--progress-seconds") {
            options.progress_seconds = parse_size(require_value(arg));
        } else if (arg == "--tokens") {
            options.tokens = parse_size(require_value(arg));
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint64_t>(std::stoull(require_value(arg)));
        } else if (arg == "--temperature") {
            options.temperature = parse_float(require_value(arg));
        } else if (arg == "--learning-rate") {
            options.learning_rate = parse_float(require_value(arg));
        } else if (arg == "--threads") {
            options.threads = parse_size(require_value(arg));
        } else if (arg == "--parallel-min-dim") {
            options.parallel_min_dimensions = parse_size(require_value(arg));
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.corpus_path.empty()) {
        throw std::runtime_error("--corpus is required");
    }
    return options;
}

std::vector<std::string> read_corpus(const Options& options) {
    std::ifstream input(options.corpus_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read corpus: " + options.corpus_path.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (options.max_lines != 0 && lines.size() >= options.max_lines) {
            break;
        }
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        if (line.size() < 24) {
            continue;
        }
        if (line.size() > options.max_line_chars) {
            line.resize(options.max_line_chars);
        }
        lines.push_back(std::move(line));
    }
    if (lines.empty()) {
        throw std::runtime_error("corpus did not contain usable lines");
    }
    return lines;
}

void print_generation_block(dzeta::OscillatorField& field,
                            std::string_view label,
                            const std::vector<std::string>& prompts,
                            std::size_t tokens) {
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        std::cout << label << "_prompt_" << (i + 1) << "=" << prompts[i] << "\n";
        std::cout << label << "_output_" << (i + 1) << "=" << field.forward(prompts[i], tokens) << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto lines = read_corpus(options);
        dzeta::OscillatorField field(options.oscillators, options.dimensions, options.seed);
        field.set_generation_temperature(options.temperature);
        field.set_learning_rate(options.learning_rate);
        field.set_thread_count(options.threads);
        field.set_parallel_min_dimensions(options.parallel_min_dimensions);
        const std::vector<std::string> prompts{
            "Once upon a time",
            "The little robot",
            "A safe assistant",
            "The child learned",
            "Open intelligence",
        };

        std::cout << "dzeta_train_smoke_begin\n";
        std::cout << "corpus_path=" << options.corpus_path.string() << "\n";
        std::cout << "corpus_lines=" << lines.size() << "\n";
        std::cout << "oscillators_limit=" << options.oscillators << "\n";
        std::cout << "dimensions=" << options.dimensions << "\n";
        std::cout << "seconds_budget=" << options.seconds << "\n";
        std::cout << "progress_seconds=" << options.progress_seconds << "\n";
        std::cout << "tokens_per_prompt=" << options.tokens << "\n";
        std::cout << "seed=" << options.seed << "\n";
        std::cout << "generation_temperature=" << static_cast<double>(options.temperature) << "\n";
        std::cout << "learning_rate=" << static_cast<double>(options.learning_rate) << "\n";
        std::cout << "threads=" << field.thread_count() << "\n";
        std::cout << "parallel_min_dimensions=" << options.parallel_min_dimensions << "\n";
        std::cout << "observations_initial=" << field.observation_count() << "\n";
        std::cout << "contrastive_updates_initial=" << field.contrastive_update_count() << "\n";
        std::cout << "mean_loss_initial=" << static_cast<double>(field.mean_loss()) << "\n";
        print_generation_block(field, "before", prompts, options.tokens);
        std::cout << std::flush;

        const auto started = std::chrono::steady_clock::now();
        const auto deadline = started + std::chrono::seconds(options.seconds);
        auto next_progress = started + std::chrono::seconds(options.progress_seconds);
        std::size_t lines_seen = 0;
        std::size_t epochs = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            for (const auto& line : lines) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    break;
                }
                field.learn(line);
                ++lines_seen;
                const auto now = std::chrono::steady_clock::now();
                if (options.progress_seconds != 0 && now >= next_progress) {
                    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - started).count();
                    std::cerr << "progress elapsed_ms=" << elapsed_ms
                              << " lines_seen=" << lines_seen
                              << " oscillators=" << field.size()
                              << " observations=" << field.observation_count()
                              << " contrastive_updates=" << field.contrastive_update_count()
                              << " mean_loss=" << static_cast<double>(field.mean_loss()) << "\n";
                    next_progress = now + std::chrono::seconds(options.progress_seconds);
                }
            }
            ++epochs;
        }

        const auto finished = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();
        const long double elapsed_seconds = static_cast<long double>(std::max<long long>(1, elapsed_ms)) / 1000.0L;
        std::cout << "elapsed_ms=" << elapsed_ms << "\n";
        std::cout << "epochs_completed=" << epochs << "\n";
        std::cout << "lines_seen=" << lines_seen << "\n";
        std::cout << "lines_per_second=" << static_cast<double>(static_cast<long double>(lines_seen) / elapsed_seconds) << "\n";
        std::cout << "oscillators_after=" << field.size() << "\n";
        std::cout << "observations_after=" << field.observation_count() << "\n";
        std::cout << "contrastive_updates_after=" << field.contrastive_update_count() << "\n";
        std::cout << "mean_loss_after=" << static_cast<double>(field.mean_loss()) << "\n";
        print_generation_block(field, "after", prompts, options.tokens);
        std::cout << "dzeta_train_smoke_end\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dzeta_train_smoke error: " << exc.what() << "\n";
        return 1;
    }
}
