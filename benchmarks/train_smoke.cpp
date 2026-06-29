#include "token_field.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
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
    std::size_t target_lines = 0;
    std::size_t max_line_chars = 512;
    std::size_t progress_seconds = 30;
    std::size_t autosave_seconds = 0;
    std::size_t tokens = 12;
    std::uint64_t seed = 0;
    long double temperature = 0.08L;
    long double learning_rate = 0.32L;
    long double update_probability = 1.0L;
    long double update_noise = 0.0L;
    long double random_init_scale = 0.0L;
    std::size_t threads = 0;
    std::size_t parallel_min_dimensions = 2048;
    bool shuffle_lines = false;
    std::filesystem::path save_model_path;
    std::filesystem::path load_model_path;
};

void print_usage() {
    std::cout
        << "Usage: dzeta_train_smoke --corpus PATH [--seconds N] [--oscillators N]\n"
        << "                         [--dimensions N] [--max-lines N] [--target-lines N]\n"
        << "                         [--max-line-chars N] [--progress-seconds N]\n"
        << "                         [--tokens N] [--seed N]\n"
        << "                         [--temperature X] [--learning-rate X]\n"
        << "                         [--threads N] [--parallel-min-dim N]\n"
        << "                         [--shuffle-lines] [--update-probability X]\n"
        << "                         [--update-noise X] [--random-init-scale X]\n"
        << "                         [--save-model PATH] [--load-model PATH]\n"
        << "                         [--autosave-seconds N]\n";
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
        } else if (arg == "--target-lines") {
            options.target_lines = parse_size(require_value(arg));
        } else if (arg == "--max-line-chars") {
            options.max_line_chars = parse_size(require_value(arg));
        } else if (arg == "--progress-seconds") {
            options.progress_seconds = parse_size(require_value(arg));
        } else if (arg == "--autosave-seconds") {
            options.autosave_seconds = parse_size(require_value(arg));
        } else if (arg == "--tokens") {
            options.tokens = parse_size(require_value(arg));
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint64_t>(std::stoull(require_value(arg)));
        } else if (arg == "--temperature") {
            options.temperature = parse_float(require_value(arg));
        } else if (arg == "--learning-rate") {
            options.learning_rate = parse_float(require_value(arg));
        } else if (arg == "--update-probability") {
            options.update_probability = parse_float(require_value(arg));
        } else if (arg == "--update-noise") {
            options.update_noise = parse_float(require_value(arg));
        } else if (arg == "--random-init-scale") {
            options.random_init_scale = parse_float(require_value(arg));
        } else if (arg == "--threads") {
            options.threads = parse_size(require_value(arg));
        } else if (arg == "--parallel-min-dim") {
            options.parallel_min_dimensions = parse_size(require_value(arg));
        } else if (arg == "--shuffle-lines") {
            options.shuffle_lines = true;
        } else if (arg == "--save-model") {
            options.save_model_path = require_value(arg);
        } else if (arg == "--load-model") {
            options.load_model_path = require_value(arg);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.corpus_path.empty()) {
        throw std::runtime_error("--corpus is required");
    }
    if (options.seconds == 0 && options.target_lines == 0) {
        throw std::runtime_error("--seconds 0 requires --target-lines N");
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

std::uint64_t entropy64() {
    std::random_device rd;
    const auto tick = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return (static_cast<std::uint64_t>(rd()) << 32U) ^
           static_cast<std::uint64_t>(rd()) ^
           (tick * 0x9e3779b97f4a7c15ULL);
}

void save_model_atomically(const dzeta::OscillatorField& field, const std::filesystem::path& path) {
    if (path.empty()) {
        return;
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    auto temporary = path;
    temporary += ".tmp";
    field.save_model(temporary.string());
    std::error_code error;
    std::filesystem::remove(path, error);
    std::filesystem::rename(temporary, path);
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
        auto lines = read_corpus(options);
        dzeta::OscillatorField field(options.oscillators, options.dimensions, options.seed);
        if (!options.load_model_path.empty()) {
            field.load_model(options.load_model_path.string());
        }
        field.set_generation_temperature(options.temperature);
        field.set_learning_rate(options.learning_rate);
        field.set_update_probability(options.update_probability);
        field.set_update_noise(options.update_noise);
        field.set_random_init_scale(options.random_init_scale);
        field.set_thread_count(options.threads);
        field.set_parallel_min_dimensions(options.parallel_min_dimensions);
        std::mt19937_64 shuffle_rng(options.seed == 0 ? entropy64() : options.seed);
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
        std::cout << "target_lines=" << options.target_lines << "\n";
        std::cout << "progress_seconds=" << options.progress_seconds << "\n";
        std::cout << "autosave_seconds=" << options.autosave_seconds << "\n";
        std::cout << "tokens_per_prompt=" << options.tokens << "\n";
        std::cout << "seed=" << options.seed << (options.seed == 0 ? " (entropy)" : "") << "\n";
        std::cout << "generation_temperature=" << static_cast<double>(options.temperature) << "\n";
        std::cout << "learning_rate=" << static_cast<double>(options.learning_rate) << "\n";
        std::cout << "update_probability=" << static_cast<double>(options.update_probability) << "\n";
        std::cout << "update_noise=" << static_cast<double>(options.update_noise) << "\n";
        std::cout << "random_init_scale=" << static_cast<double>(options.random_init_scale) << "\n";
        std::cout << "shuffle_lines=" << (options.shuffle_lines ? "true" : "false") << "\n";
        if (!options.load_model_path.empty()) {
            std::cout << "load_model_path=" << options.load_model_path.string() << "\n";
        }
        if (!options.save_model_path.empty()) {
            std::cout << "save_model_path=" << options.save_model_path.string() << "\n";
        }
        std::cout << "threads=" << field.thread_count() << "\n";
        std::cout << "parallel_min_dimensions=" << options.parallel_min_dimensions << "\n";
        std::cout << "observations_initial=" << field.observation_count() << "\n";
        std::cout << "contrastive_updates_initial=" << field.contrastive_update_count() << "\n";
        std::cout << "mean_loss_initial=" << static_cast<double>(field.mean_loss()) << "\n";
        print_generation_block(field, "before", prompts, options.tokens);
        std::cout << std::flush;

        const auto started = std::chrono::steady_clock::now();
        const auto deadline = started + std::chrono::seconds(options.seconds);
        const bool time_limited = options.seconds != 0;
        auto next_progress = started + std::chrono::seconds(options.progress_seconds);
        auto next_autosave = started + std::chrono::seconds(options.autosave_seconds);
        std::size_t lines_seen = 0;
        std::size_t epochs = 0;
        while ((!time_limited || std::chrono::steady_clock::now() < deadline) &&
               (options.target_lines == 0 || lines_seen < options.target_lines)) {
            if (options.shuffle_lines) {
                std::shuffle(lines.begin(), lines.end(), shuffle_rng);
            }
            for (const auto& line : lines) {
                if ((time_limited && std::chrono::steady_clock::now() >= deadline) ||
                    (options.target_lines != 0 && lines_seen >= options.target_lines)) {
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
                if (options.autosave_seconds != 0 && !options.save_model_path.empty() && now >= next_autosave) {
                    save_model_atomically(field, options.save_model_path);
                    std::cerr << "autosave_model=" << options.save_model_path.string()
                              << " elapsed_ms="
                              << std::chrono::duration_cast<std::chrono::milliseconds>(now - started).count()
                              << "\n";
                    next_autosave = std::chrono::steady_clock::now() +
                                    std::chrono::seconds(options.autosave_seconds);
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
        if (!options.save_model_path.empty()) {
            save_model_atomically(field, options.save_model_path);
            std::cout << "model_saved=" << options.save_model_path.string() << "\n";
        }
        print_generation_block(field, "after", prompts, options.tokens);
        std::cout << "dzeta_train_smoke_end\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dzeta_train_smoke error: " << exc.what() << "\n";
        return 1;
    }
}
