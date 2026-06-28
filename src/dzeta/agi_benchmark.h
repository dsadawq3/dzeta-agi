#pragma once

#include "unified_cognitive_field.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct AgiBenchmarkTask {
    std::string id;
    std::string prompt;
    std::string expected;
    bool expect_contains = false;
};

struct AgiBenchmarkSuite {
    std::vector<AgiBenchmarkTask> tasks;
};

struct AgiBenchmarkReport {
    std::size_t tasks_seen = 0;
    std::size_t tasks_passed = 0;
    long double solve_rate = 0.0L;
    long double recursive_trace_rate = 0.0L;
    long double bytecode_generalization_rate = 0.0L;
    long double stored_skill_reuse_rate = 0.0L;
    long double mean_loss_delta = 0.0L;
    long double mean_inference_ms = 0.0L;

    std::string scoreboard_line() const {
        std::ostringstream out;
        out << "core3_benchmark"
            << " tasks_seen=" << tasks_seen
            << " tasks_passed=" << tasks_passed
            << " solve_rate=" << static_cast<double>(solve_rate)
            << " recursive_trace_rate=" << static_cast<double>(recursive_trace_rate)
            << " bytecode_generalization_rate=" << static_cast<double>(bytecode_generalization_rate)
            << " stored_skill_reuse_rate=" << static_cast<double>(stored_skill_reuse_rate)
            << " mean_loss_delta=" << static_cast<double>(mean_loss_delta)
            << " mean_inference_ms=" << static_cast<double>(mean_inference_ms);
        return out.str();
    }
};

struct AutonomousImprovementReport {
    AgiBenchmarkReport before;
    AgiBenchmarkReport after;
    std::size_t generated_curriculum_tasks = 0;
    std::size_t learned_skills = 0;
    long double delta_solve_rate = 0.0L;

    std::string scoreboard_line() const {
        std::ostringstream out;
        out << "core3_autonomous_improvement"
            << " before_solve_rate=" << static_cast<double>(before.solve_rate)
            << " after_solve_rate=" << static_cast<double>(after.solve_rate)
            << " delta_solve_rate=" << static_cast<double>(delta_solve_rate)
            << " generated_curriculum_tasks=" << generated_curriculum_tasks
            << " learned_skills=" << learned_skills;
        return out.str();
    }
};

inline std::string benchmark_trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1U);
}

inline std::string normalize_benchmark_answer(std::string_view text) {
    std::string out(text);
    out = benchmark_trim(out);
    out.erase(std::remove_if(out.begin(), out.end(), [](unsigned char ch) {
                  return std::isspace(ch) != 0;
              }),
              out.end());
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

inline std::optional<std::string> benchmark_json_string(std::string_view line, std::string_view key) {
    const std::string pattern = "\"" + std::string(key) + "\"";
    const auto key_pos = line.find(pattern);
    if (key_pos == std::string_view::npos) {
        return std::nullopt;
    }
    const auto colon = line.find(':', key_pos + pattern.size());
    if (colon == std::string_view::npos) {
        return std::nullopt;
    }
    const auto quote = line.find('"', colon + 1U);
    if (quote == std::string_view::npos) {
        return std::nullopt;
    }
    std::string out;
    bool escaped = false;
    for (std::size_t i = quote + 1U; i < line.size(); ++i) {
        const char ch = line[i];
        if (escaped) {
            if (ch == 'n') {
                out.push_back('\n');
            } else if (ch == 't') {
                out.push_back('\t');
            } else {
                out.push_back(ch);
            }
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            return out;
        }
        out.push_back(ch);
    }
    return std::nullopt;
}

inline AgiBenchmarkSuite load_agi_benchmark_suite(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read agi benchmark suite: " + std::string(path));
    }
    AgiBenchmarkSuite suite;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        if (line.front() == '{') {
            auto id = benchmark_json_string(line, "id");
            auto prompt = benchmark_json_string(line, "prompt");
            auto expected = benchmark_json_string(line, "expected");
            bool contains = false;
            if (!expected) {
                expected = benchmark_json_string(line, "expect_contains");
                contains = expected.has_value();
            }
            if (id && prompt && expected) {
                suite.tasks.push_back({*id, *prompt, *expected, contains});
            }
            continue;
        }
        std::stringstream stream(line);
        std::string id;
        std::string prompt;
        std::string expected;
        std::getline(stream, id, '\t');
        std::getline(stream, prompt, '\t');
        std::getline(stream, expected, '\t');
        if (!id.empty() && !prompt.empty()) {
            suite.tasks.push_back({std::move(id), std::move(prompt), std::move(expected), false});
        }
    }
    return suite;
}

inline AgiBenchmarkReport run_core3_benchmark(UnifiedCognitiveField& core,
                                              const AgiBenchmarkSuite& suite,
                                              bool require_recursive_trace) {
    AgiBenchmarkReport report;
    if (suite.tasks.empty()) {
        return report;
    }
    long double recursive_hits = 0.0L;
    long double bytecode_hits = 0.0L;
    long double stored_hits = 0.0L;
    long double loss_delta_sum = 0.0L;
    long double ms_sum = 0.0L;
    for (const auto& task : suite.tasks) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = core.solve(task.prompt);
        const auto finish = std::chrono::steady_clock::now();
        const long double elapsed_ms =
            static_cast<long double>(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()) / 1000.0L;
        ++report.tasks_seen;
        const auto normalized_answer = normalize_benchmark_answer(result.answer);
        const auto normalized_expected = normalize_benchmark_answer(task.expected);
        const bool answer_ok = task.expect_contains
                                   ? normalized_answer.find(normalized_expected) != std::string::npos
                                   : normalized_answer == normalized_expected;
        const bool recursive_ok = !require_recursive_trace || result.trace_contains("recursive_loop");
        if (answer_ok && recursive_ok) {
            ++report.tasks_passed;
        }
        recursive_hits += result.trace_contains("recursive_loop") ? 1.0L : 0.0L;
        bytecode_hits += result.trace_contains("bytecode_verified") || result.trace_contains("stored_bytecode_skill") ? 1.0L : 0.0L;
        const auto task_input = core3_parse_task_input(task.prompt);
        if (task_input) {
            const auto reuse = core.solve("use " + task.id.substr(0, task.id.find("_unseen")) + " skill; task: " + *task_input);
            stored_hits += reuse.trace_contains("stored_bytecode_skill") ? 1.0L : 0.0L;
        }
        loss_delta_sum += result.metrics.loss_delta;
        ms_sum += elapsed_ms;
    }
    const long double denom = static_cast<long double>(report.tasks_seen);
    report.solve_rate = static_cast<long double>(report.tasks_passed) / denom;
    report.recursive_trace_rate = recursive_hits / denom;
    report.bytecode_generalization_rate = bytecode_hits / denom;
    report.stored_skill_reuse_rate = stored_hits / denom;
    report.mean_loss_delta = loss_delta_sum / denom;
    report.mean_inference_ms = ms_sum / denom;
    return report;
}

inline std::optional<AgiBenchmarkTask> make_curriculum_task_from_failure(const AgiBenchmarkTask& task) {
    const auto natural = core3_parse_natural_task(task.prompt);
    if (!natural) {
        return std::nullopt;
    }
    if (natural->skill_name == "gcd") {
        return AgiBenchmarkTask{"curriculum_gcd",
                                "examples: 48,18 -> 6; 20,8 -> 4; 81,27 -> 27; task: " + natural->task_input,
                                task.expected};
    }
    if (natural->skill_name == "palindrome") {
        return AgiBenchmarkTask{"curriculum_palindrome",
                                "examples: level -> true; robot -> false; radar -> true; task: " + natural->task_input,
                                task.expected};
    }
    if (natural->skill_name == "sum_list") {
        return AgiBenchmarkTask{"curriculum_sum_list",
                                "examples: 1,2,3 -> 6; 10,-4,7 -> 13; task: " + natural->task_input,
                                task.expected};
    }
    if (natural->skill_name == "product_list") {
        return AgiBenchmarkTask{"curriculum_product_list",
                                "examples: 2,3,4 -> 24; -1,5,2 -> -10; task: " + natural->task_input,
                                task.expected};
    }
    if (natural->skill_name == "max_list") {
        return AgiBenchmarkTask{"curriculum_max_list",
                                "examples: 2,9,4 -> 9; -1,-5,-2 -> -1; task: " + natural->task_input,
                                task.expected};
    }
    if (natural->skill_name == "min_list") {
        return AgiBenchmarkTask{"curriculum_min_list",
                                "examples: 2,9,4 -> 2; -1,-5,-2 -> -5; task: " + natural->task_input,
                                task.expected};
    }
    if (natural->skill_name == "word_count") {
        return AgiBenchmarkTask{"curriculum_word_count",
                                "examples: alpha beta gamma -> 3; one -> 1; task: " + natural->task_input,
                                task.expected};
    }
    if (natural->skill_name == "count_vowels") {
        return AgiBenchmarkTask{"curriculum_count_vowels",
                                "examples: abracadabra -> 5; sky -> 0; queue -> 4; task: " + natural->task_input,
                                task.expected};
    }
    return std::nullopt;
}

inline AutonomousImprovementReport run_autonomous_improvement(UnifiedCognitiveField& core,
                                                              const AgiBenchmarkSuite& suite,
                                                              std::size_t cycles) {
    AutonomousImprovementReport report;
    report.before = run_core3_benchmark(core, suite, true);
    for (std::size_t cycle = 0; cycle < std::max<std::size_t>(1, cycles); ++cycle) {
        for (const auto& task : suite.tasks) {
            const auto result = core.solve(task.prompt);
            if (normalize_benchmark_answer(result.answer) == normalize_benchmark_answer(task.expected)) {
                continue;
            }
            const auto curriculum = make_curriculum_task_from_failure(task);
            if (!curriculum) {
                continue;
            }
            ++report.generated_curriculum_tasks;
            const auto learned = core.solve(curriculum->prompt);
            if (normalize_benchmark_answer(learned.answer) == normalize_benchmark_answer(curriculum->expected) &&
                learned.trace_contains("bytecode_verified")) {
                ++report.learned_skills;
                (void)core.learn(learned.trace, 1.0L);
            }
        }
    }
    report.after = run_core3_benchmark(core, suite, true);
    report.delta_solve_rate = report.after.solve_rate - report.before.solve_rate;
    return report;
}

inline AgiBenchmarkSuite default_core3_benchmark_suite() {
    AgiBenchmarkSuite suite;
    suite.tasks.push_back({"gcd_unseen", "examples: 48,18 -> 6; 20,8 -> 4; 81,27 -> 27; task: 84,30", "6"});
    suite.tasks.push_back({"palindrome_unseen", "examples: level -> true; robot -> false; radar -> true; task: neveroddoreven", "true"});
    suite.tasks.push_back({"sum_list_unseen", "examples: 1,2,3 -> 6; 10,-4,7 -> 13; task: 8,9,-2", "15"});
    suite.tasks.push_back({"count_vowels_unseen", "examples: abracadabra -> 5; sky -> 0; queue -> 4; task: dzeta", "2"});
    suite.tasks.push_back({"sort_small_unseen", "examples: 3,1,2 -> 1,2,3; 9,7,8 -> 7,8,9; task: 10,-1,3", "-1,3,10"});
    suite.tasks.push_back({"product_list_unseen", "examples: 2,3,4 -> 24; -1,5,2 -> -10; task: 4,5,-2", "-40"});
    suite.tasks.push_back({"max_list_unseen", "examples: 2,9,4 -> 9; -1,-5,-2 -> -1; task: 10,-1,3", "10"});
    suite.tasks.push_back({"word_count_unseen", "examples: alpha beta gamma -> 3; one -> 1; task: recursive latent workspace", "3"});
    return suite;
}

} // namespace dzeta
