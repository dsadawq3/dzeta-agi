#include "token_field.h"

#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Behavioral positional-transfer regression: a continuation trained at
// several line offsets must fire from a prompt that places the same marker
// context at a NOVEL offset, and the repeated n-gram must merge into one
// context prototype instead of one per offset. Under the old
// absolute-position waves both properties failed by construction.
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

int count_hits(const std::string& output, const std::set<std::string>& targets) {
    int hits = 0;
    for (const auto& word : split_words(output)) {
        if (targets.count(word) != 0) {
            ++hits;
        }
    }
    return hits;
}

void check(long double interference) {
    const std::vector<std::string> corpus{
        "the silver comet streaks across the night sky over quiet hills",
        "children waved when the silver comet streaks across the night sky",
        "yesterday evening people near the tower saw the silver comet streaks across the night sky",
        "a historian compares archives and old treaties in the library",
        "the chef prepares warm soup with careful spices tonight",
        "a telescope tracks distant planets beyond the cold mountains",
    };
    dzeta::OscillatorField field(8192, 512, 20260726);
    field.set_generation_temperature(0.0L);
    field.set_learning_rate(1.0L);
    field.set_thread_count(4);
    field.set_parallel_min_dimensions(1);
    field.set_dimension_interference(interference);
    for (int pass = 0; pass < 3; ++pass) {
        for (const auto& line : corpus) {
            field.learn(line);
        }
    }

    const std::set<std::string> targets{"across", "night", "sky"};
    // Marker at a trained offset.
    assert(count_hits(field.forward("the silver comet streaks", 8), targets) >= 2);
    // Marker at a NOVEL offset behind a novel prefix.
    assert(count_hits(field.forward("yesterday evening we watched the silver comet streaks", 8),
                      targets) >= 2);

    // Sample-efficiency guarantee: 3 offsets x 3 passes of the same n-gram
    // context land in one (or at most two) prototypes, not one per offset.
    bool found_across = false;
    for (const auto& summary : field.token_summaries(0)) {
        if (summary.token == "across") {
            found_across = true;
            assert(summary.observations == 9);
            assert(summary.prototypes <= 2);
        }
    }
    assert(found_across);

    // Long-generation guard: past all trained line lengths the output must
    // stay populated and lexically diverse (the old representation went
    // statistically independent of every learned key here).
    const auto words = split_words(field.forward("the silver comet streaks", 48));
    assert(words.size() >= 40);
    std::map<std::string, int> counts;
    for (const auto& word : words) {
        ++counts[word];
    }
    assert(counts.size() >= 12);
}

}  // namespace

int main() {
    check(0.0L);
    check(0.25L);
    std::cout << "dzeta_translation_transfer passed\n";
    return 0;
}
