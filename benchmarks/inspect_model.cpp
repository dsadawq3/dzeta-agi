#include "token_field.h"

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path model_path;
    std::size_t top = 20;
    std::size_t tokens = 16;
    std::vector<std::string> inspect_tokens;
    std::vector<std::string> prompts;
};

std::size_t parse_size(std::string_view value) {
    if (value.empty()) {
        throw std::runtime_error("empty numeric value");
    }
    return static_cast<std::size_t>(std::stoull(std::string(value)));
}

void print_usage() {
    std::cout
        << "Usage: dzeta_inspect_model --model PATH [--top N]\n"
        << "                           [--token WORD ...] [--prompt TEXT ...]\n"
        << "                           [--tokens N]\n";
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
        if (arg == "--model") {
            options.model_path = require_value(arg);
        } else if (arg == "--top") {
            options.top = parse_size(require_value(arg));
        } else if (arg == "--tokens") {
            options.tokens = parse_size(require_value(arg));
        } else if (arg == "--token") {
            options.inspect_tokens.emplace_back(require_value(arg));
        } else if (arg == "--prompt") {
            options.prompts.emplace_back(require_value(arg));
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (options.model_path.empty()) {
        throw std::runtime_error("--model is required");
    }
    return options;
}

void print_top_tokens(const dzeta::OscillatorField& field, std::size_t top) {
    std::cout << "top_tokens_begin\n";
    std::cout << "rank\ttoken\tobservations\tstrength\terror_ema\tprototypes\n";
    const auto summaries = field.token_summaries(top);
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        const auto& summary = summaries[i];
        std::cout << (i + 1) << '\t'
                  << summary.token << '\t'
                  << summary.observations << '\t'
                  << static_cast<double>(summary.strength) << '\t'
                  << static_cast<double>(summary.error_ema) << '\t'
                  << summary.prototypes << '\n';
    }
    std::cout << "top_tokens_end\n";
}

void print_token_links(const dzeta::OscillatorField& field,
                       std::string_view token,
                       std::size_t top) {
    std::cout << "token_links_begin token=" << token << "\n";
    std::cout << "rank\ttoken\tscore\tnext_sim\tcontext_sim\ttransition_sim\tpadic_sim\tobservations\n";
    const auto links = field.nearest_token_links(token, top);
    for (std::size_t i = 0; i < links.size(); ++i) {
        const auto& link = links[i];
        std::cout << (i + 1) << '\t'
                  << link.token << '\t'
                  << static_cast<double>(link.association_score) << '\t'
                  << static_cast<double>(link.next_similarity) << '\t'
                  << static_cast<double>(link.context_similarity) << '\t'
                  << static_cast<double>(link.transition_similarity) << '\t'
                  << static_cast<double>(link.padic_similarity) << '\t'
                  << link.observations << '\n';
    }
    if (links.empty()) {
        std::cout << "no_links\n";
    }
    std::cout << "token_links_end token=" << token << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        dzeta::OscillatorField field;
        field.load_model(options.model_path.string());

        std::cout << "dzeta_inspect_model_begin\n";
        std::cout << "model_path=" << options.model_path.string() << "\n";
        std::cout << "oscillators=" << field.size() << "\n";
        std::cout << "oscillator_limit=" << field.oscillator_limit() << "\n";
        std::cout << "dimensions=" << field.dimensions() << "\n";
        std::cout << "observations=" << field.observation_count() << "\n";
        std::cout << "contrastive_updates=" << field.contrastive_update_count() << "\n";
        std::cout << "mean_loss=" << static_cast<double>(field.mean_loss()) << "\n";

        print_top_tokens(field, options.top);
        for (const auto& token : options.inspect_tokens) {
            print_token_links(field, token, options.top);
        }
        for (const auto& prompt : options.prompts) {
            std::cout << "prompt=" << prompt << "\n";
            std::cout << "prompt_output=" << field.forward(prompt, options.tokens) << "\n";
        }

        std::cout << "dzeta_inspect_model_end\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dzeta_inspect_model error: " << exc.what() << "\n";
        return 1;
    }
}
