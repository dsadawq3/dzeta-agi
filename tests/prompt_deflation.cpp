#include "token_field.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::set<std::string> words(const std::string& text) {
    std::string normalized;
    normalized.reserve(text.size());
    for (unsigned char ch : text) {
        if (std::isalnum(ch)) {
            normalized.push_back(static_cast<char>(std::tolower(ch)));
        } else {
            normalized.push_back(' ');
        }
    }
    std::istringstream input(normalized);
    std::set<std::string> out;
    std::string token;
    while (input >> token) {
        if (token.size() > 1) {
            out.insert(token);
        }
    }
    return out;
}

long double pair_overlap(const std::set<std::string>& left, const std::set<std::string>& right) {
    if (left.empty() || right.empty()) {
        return 0.0L;
    }
    std::size_t shared = 0;
    for (const auto& token : left) {
        if (right.find(token) != right.end()) {
            ++shared;
        }
    }
    return static_cast<long double>(shared) /
           static_cast<long double>(std::min(left.size(), right.size()));
}

long double mean_overlap(const std::vector<std::string>& outputs) {
    std::vector<std::set<std::string>> token_sets;
    token_sets.reserve(outputs.size());
    for (const auto& output : outputs) {
        token_sets.push_back(words(output));
    }
    long double total = 0.0L;
    std::size_t pairs = 0;
    for (std::size_t i = 0; i < token_sets.size(); ++i) {
        for (std::size_t j = i + 1; j < token_sets.size(); ++j) {
            total += pair_overlap(token_sets[i], token_sets[j]);
            ++pairs;
        }
    }
    return pairs == 0 ? 0.0L : total / static_cast<long double>(pairs);
}

dzeta::OscillatorField trained_field(long double dim_interference) {
    const std::vector<std::string> corpus{
        "common bright garden path everyone carefully follows the same beautiful pattern while the princess finds a hidden crown",
        "common bright garden path everyone carefully follows the same beautiful pattern while the robot repairs a silver sensor",
        "common bright garden path everyone carefully follows the same beautiful pattern while the assistant explains a safe medical warning",
        "common bright garden path everyone carefully follows the same beautiful pattern while the child learns fractions from colored blocks",
        "common bright garden path everyone carefully follows the same beautiful pattern while open intelligence shares transparent research notes",
        "common bright garden path everyone carefully follows the same beautiful pattern while a telescope tracks planets and comet dust",
        "common bright garden path everyone carefully follows the same beautiful pattern while a chef balances spices in a quiet kitchen",
        "common bright garden path everyone carefully follows the same beautiful pattern while a historian compares archives and treaties",
        "common bright garden path everyone carefully follows the same beautiful pattern while the robot calibrates motors and checks a battery",
        "common bright garden path everyone carefully follows the same beautiful pattern while the robot maps a hallway with sonar",
        "common bright garden path everyone carefully follows the same beautiful pattern while the assistant warns about allergies and dosage",
        "common bright garden path everyone carefully follows the same beautiful pattern while the assistant summarizes instructions for a patient",
        "common bright garden path everyone carefully follows the same beautiful pattern while the child solves fractions with patient practice",
        "common bright garden path everyone carefully follows the same beautiful pattern while the child reads a lesson about numbers",
        "common bright garden path everyone carefully follows the same beautiful pattern while open intelligence publishes reproducible code",
        "common bright garden path everyone carefully follows the same beautiful pattern while open intelligence audits safety claims",
        "common bright garden path everyone carefully follows the same beautiful pattern while a princess returns a crown to the village",
        "common bright garden path everyone carefully follows the same beautiful pattern while a castle guard unlocks a moonlit gate",
        "common bright garden path everyone carefully follows the same beautiful pattern while a telescope measures a distant orbit",
        "common bright garden path everyone carefully follows the same beautiful pattern while a comet leaves silver dust behind",
        "common bright garden path everyone carefully follows the same beautiful pattern while a chef prepares soup with careful spices",
        "common bright garden path everyone carefully follows the same beautiful pattern while a kitchen timer rings before dinner",
        "common bright garden path everyone carefully follows the same beautiful pattern while a historian restores maps from an archive",
        "common bright garden path everyone carefully follows the same beautiful pattern while treaties explain old borders and trade",
    };

    dzeta::OscillatorField field(8192, 768, 424242);
    field.set_generation_temperature(0.0L);
    field.set_learning_rate(1.0L);
    field.set_thread_count(4);
    field.set_parallel_min_dimensions(1);
    field.set_dimension_interference(dim_interference);

    for (int pass = 0; pass < 3; ++pass) {
        for (const auto& line : corpus) {
            field.learn(line);
        }
    }
    return field;
}

std::vector<std::string> outputs_for(dzeta::OscillatorField& field) {
    const std::vector<std::string> prompts{
        "Once upon a time",
        "The little robot",
        "A safe assistant",
        "The child learned",
        "Open intelligence",
    };
    std::vector<std::string> outputs;
    outputs.reserve(prompts.size());
    for (const auto& prompt : prompts) {
        outputs.push_back(field.forward(prompt, 10));
    }
    return outputs;
}

}  // namespace

int main() {
    auto baseline = trained_field(0.0L);
    auto experimental = trained_field(0.25L);

    const auto baseline_outputs = outputs_for(baseline);
    const auto experimental_outputs = outputs_for(experimental);
    const long double baseline_overlap = mean_overlap(baseline_outputs);
    const long double experimental_overlap = mean_overlap(experimental_outputs);

    std::cout << "baseline_overlap=" << static_cast<double>(baseline_overlap) << "\n";
    std::cout << "experimental_overlap=" << static_cast<double>(experimental_overlap) << "\n";
    std::cout << std::flush;

    assert(baseline_overlap > 0.70L);
    assert(experimental_overlap < 0.35L);

    std::cout << "dzeta_prompt_deflation passed\n";
    return 0;
}
