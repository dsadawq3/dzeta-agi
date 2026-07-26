#include "token_field.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Long-generation loop control: periodic cycles and window-scale repetition
// must stay bounded far past trained line lengths, and long outputs must be
// bit-identical across thread counts (the regime where the cycle-penalty and
// rollout-ban code paths are actually hot).
namespace {

std::vector<std::string> split_words(const std::string& text) {
    std::istringstream input(text);
    std::vector<std::string> out;
    std::string token;
    while (input >> token) {
        out.push_back(token);
    }
    return out;
}

std::size_t max_periodic_run(const std::vector<std::string>& words, std::size_t period) {
    std::size_t best = 0;
    std::size_t run = 0;
    for (std::size_t i = 0; i + period < words.size(); ++i) {
        if (words[i] == words[i + period]) {
            ++run;
            best = std::max(best, run);
        } else {
            run = 0;
        }
    }
    return best;
}

std::size_t max_window_occurrences(const std::vector<std::string>& words) {
    std::size_t worst = 0;
    for (std::size_t begin = 0; begin < words.size(); ++begin) {
        const std::size_t end = std::min(words.size(), begin + 32);
        std::map<std::string, std::size_t> counts;
        for (std::size_t i = begin; i < end; ++i) {
            if (words[i].size() > 3) {
                worst = std::max(worst, ++counts[words[i]]);
            }
        }
    }
    return worst;
}

dzeta::OscillatorField trained_field(long double interference, std::size_t threads) {
    const std::vector<std::string> corpus{
        "the silver comet streaks across the night sky over quiet hills",
        "children waved when the silver comet streaks across the night sky",
        "a historian compares archives and old treaties in the library",
        "the chef prepares warm soup with careful spices tonight",
        "a telescope tracks distant planets beyond the cold mountains",
        "the child learns fractions from colored blocks after school",
        "open intelligence publishes reproducible code and safety notes",
        "a princess returns a hidden crown to the village gate",
    };
    dzeta::OscillatorField field(8192, 256, 1337);
    field.set_generation_temperature(0.0L);
    field.set_learning_rate(1.0L);
    field.set_thread_count(threads);
    field.set_parallel_min_dimensions(1);
    field.set_dimension_interference(interference);
    for (int pass = 0; pass < 3; ++pass) {
        for (const auto& line : corpus) {
            field.learn(line);
        }
    }
    return field;
}

void check(long double interference) {
    auto field = trained_field(interference, 4);
    auto serial = trained_field(interference, 1);
    const std::vector<std::string> prompts{
        "the silver comet streaks",
        "a historian compares",
        "open intelligence",
    };
    for (const auto& prompt : prompts) {
        const auto output = field.forward(prompt, 48);
        const auto words = split_words(output);
        assert(words.size() >= 32);
        // (a) periodic cycles must stay bounded
        assert(max_periodic_run(words, 2) <= 6);
        assert(max_periodic_run(words, 3) <= 9);
        // (b) no content word floods a 32-token window
        assert(max_window_occurrences(words) <= 5);
        // (c) lexical diversity over the long output
        std::map<std::string, int> counts;
        for (const auto& word : words) {
            ++counts[word];
        }
        assert(counts.size() * 100 >= words.size() * 35);
        // (d) long-generation determinism across thread counts
        assert(serial.forward(prompt, 48) == output);
    }
}

}  // namespace

int main() {
    check(0.0L);
    check(0.25L);
    std::cout << "dzeta_repetition passed\n";
    return 0;
}
