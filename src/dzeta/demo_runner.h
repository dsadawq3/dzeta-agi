#pragma once

#include "field_reasoning.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct DemoTask {
    std::string id;
    std::string type;
    std::string prompt;
    std::string expect_contains;
};

struct DemoTaskResult {
    DemoTask task;
    FieldReasoningResult reasoning;
    bool passed = false;
    bool code_valid = false;
    long double confidence = 0.0L;
    long double analogy_score = 0.0L;
    long double causal_score = 0.0L;
};

struct DemoSuiteReport {
    std::size_t tasks_seen = 0;
    std::size_t tasks_passed = 0;
    long double code_validity_score = 0.0L;
    long double analogy_transfer_score = 0.0L;
    long double causal_intervention_score = 0.0L;
    long double mean_confidence = 0.0L;
    std::vector<DemoTaskResult> results;
};

inline std::string json_unescape(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool escaped = false;
    for (char ch : text) {
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
        out.push_back(ch);
    }
    if (escaped) {
        out.push_back('\\');
    }
    return out;
}

inline std::string json_string_field(std::string_view line, std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    const std::size_t key_pos = line.find(needle);
    if (key_pos == std::string_view::npos) {
        return {};
    }
    const std::size_t colon = line.find(':', key_pos + needle.size());
    if (colon == std::string_view::npos) {
        return {};
    }
    const std::size_t begin_quote = line.find('"', colon + 1U);
    if (begin_quote == std::string_view::npos) {
        return {};
    }
    std::size_t end_quote = begin_quote + 1U;
    bool escaped = false;
    for (; end_quote < line.size(); ++end_quote) {
        const char ch = line[end_quote];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            break;
        }
    }
    if (end_quote >= line.size()) {
        return {};
    }
    return json_unescape(line.substr(begin_quote + 1U, end_quote - begin_quote - 1U));
}

inline std::vector<DemoTask> load_demo_tasks(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read demo suite: " + std::string(path));
    }
    std::vector<DemoTask> tasks;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        DemoTask task;
        task.id = json_string_field(line, "id");
        task.type = json_string_field(line, "type");
        task.prompt = json_string_field(line, "prompt");
        task.expect_contains = json_string_field(line, "expect_contains");
        if (!task.prompt.empty()) {
            if (task.id.empty()) {
                task.id = "task_" + std::to_string(tasks.size() + 1U);
            }
            if (task.type.empty()) {
                task.type = "natural";
            }
            tasks.push_back(std::move(task));
        }
    }
    return tasks;
}

inline bool answer_contains_expected(std::string_view answer, std::string_view expected) {
    if (expected.empty()) {
        return !answer.empty();
    }
    const auto lower_answer = field_reasoning_lower_copy(answer);
    const auto lower_expected = field_reasoning_lower_copy(expected);
    return lower_answer.find(lower_expected) != std::string::npos;
}

inline long double candidate_kind_score(const FieldReasoningResult& result, std::string_view kind) {
    long double score = 0.0L;
    for (const auto& candidate : result.candidates) {
        if (candidate.kind == kind) {
            score = std::max(score, candidate.score);
        }
    }
    return std::clamp(score, 0.0L, 1.0L);
}

inline DemoSuiteReport run_demo_suite(std::string_view path,
                                      IuttEnsemble& ensemble,
                                      const CodeTokenMemory& token_memory,
                                      SemanticFieldMemory& semantic_memory,
                                      const FieldReasoner& reasoner,
                                      bool verify_code) {
    const auto tasks = load_demo_tasks(path);
    DemoSuiteReport report;
    report.results.reserve(tasks.size());

    long double confidence_sum = 0.0L;
    long double code_sum = 0.0L;
    std::size_t code_tasks = 0;
    long double analogy_sum = 0.0L;
    std::size_t analogy_tasks = 0;
    long double causal_sum = 0.0L;
    std::size_t causal_tasks = 0;

    for (const auto& task : tasks) {
        DemoTaskResult result;
        result.task = task;
        const bool generate_code = task.type == "python" || detect_phase_language_mode(task.prompt) == PhaseLanguageMode::Python;
        result.reasoning = reasoner.reason(task.prompt, ensemble, token_memory, semantic_memory, generate_code);
        result.confidence = result.reasoning.best_score;
        result.code_valid = !generate_code || !verify_code ||
                            phase_language_accepts(result.reasoning.answer, PhaseLanguageMode::Python);
        result.analogy_score = candidate_kind_score(result.reasoning, "analogy_field");
        result.causal_score = candidate_kind_score(result.reasoning, "causal_field");
        result.passed = answer_contains_expected(result.reasoning.answer, task.expect_contains) && result.code_valid;

        ++report.tasks_seen;
        if (result.passed) {
            ++report.tasks_passed;
        }
        confidence_sum += result.confidence;
        if (generate_code) {
            ++code_tasks;
            code_sum += result.code_valid ? 1.0L : 0.0L;
        }
        if (task.type == "analogy") {
            ++analogy_tasks;
            analogy_sum += result.analogy_score;
        }
        if (task.type == "causal") {
            ++causal_tasks;
            causal_sum += result.causal_score;
        }
        report.results.push_back(std::move(result));
    }

    if (report.tasks_seen != 0) {
        report.mean_confidence = confidence_sum / static_cast<long double>(report.tasks_seen);
    }
    if (code_tasks != 0) {
        report.code_validity_score = code_sum / static_cast<long double>(code_tasks);
    }
    if (analogy_tasks != 0) {
        report.analogy_transfer_score = analogy_sum / static_cast<long double>(analogy_tasks);
    }
    if (causal_tasks != 0) {
        report.causal_intervention_score = causal_sum / static_cast<long double>(causal_tasks);
    }
    return report;
}

inline std::string demo_suite_summary(const DemoSuiteReport& report) {
    std::ostringstream out;
    out << "scoreboard tasks=" << report.tasks_seen
        << " passed=" << report.tasks_passed
        << " code_validity=" << static_cast<double>(report.code_validity_score)
        << " analogy_transfer=" << static_cast<double>(report.analogy_transfer_score)
        << " causal_intervention=" << static_cast<double>(report.causal_intervention_score)
        << " mean_confidence=" << static_cast<double>(report.mean_confidence);
    return out.str();
}

} // namespace dzeta
