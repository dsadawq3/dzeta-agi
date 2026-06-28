#pragma once

#include "sat.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct PythonCodeVerification {
    bool ok = false;
    std::vector<std::string> undefined_symbols;
    std::vector<std::string> issues;
};

inline bool code_verifier_builtin(std::string_view symbol) {
    static const std::set<std::string> builtins{
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "len", "list", "max", "min", "pow", "range", "reversed", "round",
        "set", "sorted", "str", "sum", "tuple", "zip", "true", "false",
        "none", "valueerror", "runtimeerror", "keyerror", "typeerror",
        "items", "keys", "values", "get", "setdefault", "append", "extend",
        "pop", "add", "discard", "remove", "update", "difference", "difference_update",
        "insert", "reverse", "sort", "heapq", "heappop", "heappush", "deque", "popleft"};
    return builtins.find(std::string(symbol)) != builtins.end();
}

inline std::string code_verifier_trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1U);
}

inline std::string code_verifier_function_name(std::string_view code) {
    const auto def = code.find("def ");
    if (def == std::string_view::npos) {
        return {};
    }
    std::size_t pos = def + 4U;
    std::string out;
    while (pos < code.size()) {
        const unsigned char ch = static_cast<unsigned char>(code[pos]);
        if (std::isalnum(ch) == 0 && code[pos] != '_') {
            break;
        }
        out.push_back(static_cast<char>(std::tolower(ch)));
        ++pos;
    }
    return out;
}

inline std::vector<std::string> code_verifier_parameters(std::string_view code) {
    std::vector<std::string> out;
    const auto def = code.find("def ");
    if (def == std::string_view::npos) {
        return out;
    }
    const auto open = code.find('(', def);
    const auto close = code.find(')', open == std::string_view::npos ? def : open);
    if (open == std::string_view::npos || close == std::string_view::npos || close <= open) {
        return out;
    }
    std::stringstream stream(std::string(code.substr(open + 1U, close - open - 1U)));
    std::string item;
    while (std::getline(stream, item, ',')) {
        item = code_verifier_trim(std::move(item));
        const auto colon = item.find(':');
        if (colon != std::string::npos) {
            item.resize(colon);
        }
        const auto equals = item.find('=');
        if (equals != std::string::npos) {
            item.resize(equals);
        }
        item = code_verifier_trim(std::move(item));
        while (!item.empty() && (item.front() == '*' || item.front() == '/')) {
            item.erase(item.begin());
        }
        item = code_verifier_trim(std::move(item));
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

inline void code_verifier_add_unique(std::vector<std::string>& values, std::string value) {
    if (!value.empty() && std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(std::move(value));
    }
}

inline std::size_t code_verifier_assignment_pos(std::string_view text) {
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '=') {
            continue;
        }
        const char prev = i == 0 ? '\0' : text[i - 1U];
        const char next = i + 1U >= text.size() ? '\0' : text[i + 1U];
        if (prev == '=' || prev == '<' || prev == '>' || prev == '!' || next == '=') {
            continue;
        }
        return i;
    }
    return std::string_view::npos;
}

inline void code_verifier_add_binding_tokens(std::set<std::string>& defined,
                                             std::string_view text) {
    for (auto token : tokenize_query(text)) {
        if (token == "for" || token == "in") {
            continue;
        }
        defined.insert(std::move(token));
    }
}

inline void code_verifier_collect_comprehension_bindings(std::set<std::string>& defined,
                                                         std::string_view text) {
    std::size_t search = 0;
    while (search < text.size()) {
        const auto for_pos = text.find(" for ", search);
        if (for_pos == std::string_view::npos) {
            break;
        }
        const auto in_pos = text.find(" in ", for_pos + 5U);
        if (in_pos == std::string_view::npos) {
            break;
        }
        code_verifier_add_binding_tokens(defined, text.substr(for_pos + 5U, in_pos - for_pos - 5U));
        search = in_pos + 4U;
    }
}

inline void code_verifier_collect_import_bindings(std::set<std::string>& defined,
                                                  std::string_view text) {
    auto trimmed = code_verifier_trim(std::string(text));
    auto define_import_item = [&](std::string item) {
        item = code_verifier_trim(std::move(item));
        const auto as_pos = item.find(" as ");
        if (as_pos != std::string::npos) {
            item = code_verifier_trim(item.substr(as_pos + 4U));
        } else {
            const auto dot = item.find('.');
            if (dot != std::string::npos) {
                item.resize(dot);
            }
        }
        for (auto token : tokenize_query(item)) {
            if (!token.empty() && token != "as") {
                defined.insert(std::move(token));
                break;
            }
        }
    };
    if (trimmed.rfind("import ", 0) == 0) {
        std::stringstream stream(trimmed.substr(7U));
        std::string item;
        while (std::getline(stream, item, ',')) {
            define_import_item(std::move(item));
        }
        return;
    }
    if (trimmed.rfind("from ", 0) == 0) {
        const auto import_pos = trimmed.find(" import ");
        if (import_pos == std::string::npos) {
            return;
        }
        std::stringstream stream(trimmed.substr(import_pos + 8U));
        std::string item;
        while (std::getline(stream, item, ',')) {
            define_import_item(std::move(item));
        }
    }
}

inline std::string code_verifier_strip_string_literals(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_single = false;
    bool in_double = false;
    bool escape = false;
    for (const char ch : text) {
        if (escape) {
            escape = false;
            continue;
        }
        if ((in_single || in_double) && ch == '\\') {
            escape = true;
            continue;
        }
        if (!in_double && ch == '\'') {
            in_single = !in_single;
            out.push_back(' ');
            continue;
        }
        if (!in_single && ch == '"') {
            in_double = !in_double;
            out.push_back(' ');
            continue;
        }
        if (!in_single && !in_double) {
            out.push_back(ch);
        } else if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            out.push_back(' ');
        }
    }
    return out;
}

inline std::optional<std::string> code_verifier_leading_triple_quote(std::string_view text) {
    std::size_t pos = 0;
    while (pos < text.size()) {
        const char ch = text[pos];
        if (ch == 'r' || ch == 'R' || ch == 'u' || ch == 'U' ||
            ch == 'f' || ch == 'F' || ch == 'b' || ch == 'B') {
            ++pos;
            continue;
        }
        break;
    }
    if (text.substr(pos, 3U) == "\"\"\"") {
        return std::string("\"\"\"");
    }
    if (text.substr(pos, 3U) == "'''") {
        return std::string("'''");
    }
    return std::nullopt;
}

inline PythonCodeVerification verify_python_function_code(std::string_view code) {
    PythonCodeVerification report;
    if (code.find("def ") == std::string_view::npos) {
        report.issues.push_back("missing_def");
        return report;
    }
    if (code.find(':') == std::string_view::npos) {
        report.issues.push_back("missing_colon");
        return report;
    }
    if (code.find("return ") == std::string_view::npos &&
        code.find("yield ") == std::string_view::npos &&
        code.find("pass") == std::string_view::npos) {
        report.issues.push_back("missing_body_result");
        return report;
    }

    std::set<std::string> defined;
    const auto function_name = code_verifier_function_name(code);
    if (!function_name.empty()) {
        defined.insert(function_name);
        for (auto token : tokenize_query(function_name)) {
            defined.insert(std::move(token));
        }
    }
    for (auto parameter : code_verifier_parameters(code)) {
        std::transform(parameter.begin(), parameter.end(), parameter.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        defined.insert(std::move(parameter));
    }

    std::stringstream stream{std::string(code)};
    std::string line;
    std::string executable_code;
    bool in_docstring = false;
    std::string docstring_delimiter;
    while (std::getline(stream, line)) {
        auto trimmed = code_verifier_trim(line);
        if (trimmed.empty() || trimmed.rfind("#", 0) == 0) {
            continue;
        }
        if (in_docstring) {
            if (trimmed.find(docstring_delimiter) != std::string::npos) {
                in_docstring = false;
                docstring_delimiter.clear();
            }
            continue;
        }
        if (const auto delimiter = code_verifier_leading_triple_quote(trimmed)) {
            const auto open = trimmed.find(*delimiter);
            const auto close = trimmed.find(*delimiter, open + delimiter->size());
            if (close == std::string::npos) {
                in_docstring = true;
                docstring_delimiter = *delimiter;
            }
            continue;
        }
        const auto comment = trimmed.find('#');
        if (comment != std::string::npos) {
            trimmed = code_verifier_trim(trimmed.substr(0, comment));
            if (trimmed.empty()) {
                continue;
            }
        }
        if (trimmed.rfind("def ", 0) == 0) {
            const auto nested_function_name = code_verifier_function_name(trimmed);
            if (!nested_function_name.empty()) {
                defined.insert(nested_function_name);
            }
            for (auto parameter : code_verifier_parameters(trimmed)) {
                std::transform(parameter.begin(), parameter.end(), parameter.begin(), [](unsigned char ch) {
                    return static_cast<char>(std::tolower(ch));
                });
                defined.insert(std::move(parameter));
            }
        } else if (trimmed.rfind("import ", 0) == 0 || trimmed.rfind("from ", 0) == 0) {
            code_verifier_collect_import_bindings(defined, trimmed);
            executable_code += code_verifier_strip_string_literals(trimmed);
            executable_code.push_back('\n');
        } else {
            executable_code += code_verifier_strip_string_literals(trimmed);
            executable_code.push_back('\n');
        }
        if (trimmed.rfind("for ", 0) == 0) {
            const auto in_pos = trimmed.find(" in ");
            if (in_pos != std::string::npos) {
                auto left = trimmed.substr(4U, in_pos - 4U);
                code_verifier_add_binding_tokens(defined, left);
            }
        }
        code_verifier_collect_comprehension_bindings(defined, trimmed);
        const auto assign = code_verifier_assignment_pos(trimmed);
        if (assign != std::string::npos) {
            auto left = code_verifier_trim(trimmed.substr(0, assign));
            if (!left.empty() && left.find('.') == std::string::npos) {
                code_verifier_add_binding_tokens(defined, left);
            }
        }
    }

    const auto tokens = tokenize_query(executable_code);
    std::string previous;
    for (const auto& raw : tokens) {
        auto token = raw;
        if (token.empty() || token == "def" || token == "return" || token == "yield" ||
            token == "if" || token == "else" || token == "elif" || token == "for" ||
            token == "while" || token == "in" || token == "and" || token == "or" ||
            token == "not" || token == "is" || token == "as" || token == "raise" ||
            token == "try" || token == "except" || token == "finally" || token == "with" ||
            token == "class" || token == "from" || token == "import" || token == "lambda" ||
            token == "assert" || token == "break" || token == "continue" || token == "del") {
            previous = token;
            continue;
        }
        if (previous == "def" || previous == "class" || previous == "import" || previous == "from") {
            previous = token;
            continue;
        }
        if (!token.empty() && std::isdigit(static_cast<unsigned char>(token.front())) != 0) {
            previous = token;
            continue;
        }
        if (defined.find(token) == defined.end() && !code_verifier_builtin(token)) {
            code_verifier_add_unique(report.undefined_symbols, token);
        }
        previous = token;
    }

    report.ok = report.undefined_symbols.empty() && report.issues.empty();
    return report;
}

inline std::string code_verifier_lower(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

inline PythonCodeVerification verify_python_contract_against_prompt(std::string_view prompt,
                                                                    std::string_view code) {
    PythonCodeVerification report;
    report.ok = true;
    const auto lower_prompt = code_verifier_lower(prompt);
    const auto lower_code = code_verifier_lower(code);
    const auto has_prompt = [&](std::string_view token) {
        return lower_prompt.find(token) != std::string::npos;
    };
    const auto has_code = [&](std::string_view token) {
        return lower_code.find(token) != std::string::npos;
    };
    const auto fail = [&](std::string issue) {
        report.ok = false;
        code_verifier_add_unique(report.issues, std::move(issue));
    };

    const bool breadth_first_prompt =
        has_prompt("bfs") || has_prompt("breadth") ||
        (has_prompt("unweighted") && has_prompt("path"));

    if (has_prompt("dijkstra") ||
        ((has_prompt("shortest") && has_prompt("path")) && !breadth_first_prompt)) {
        if (!(has_code("distances") && has_code("previous") &&
              (has_code("heapq.heappop") || has_code("frontier.sort")) &&
              has_code("for neighbor") && has_code("return path"))) {
            fail("missing_dijkstra_invariants");
        }
    }
    if (breadth_first_prompt) {
        if (!(has_code("frontier") && has_code("pop(0)") && has_code("visited") &&
              has_code("previous") && has_code("for neighbor") && has_code("return path"))) {
            fail("missing_bfs_invariants");
        }
    }
    if (has_prompt("topological") || (has_prompt("dependency") && has_prompt("graph"))) {
        if (!(has_code("indegree") && has_code("ready") && has_code("raise valueerror"))) {
            fail("missing_topological_invariants");
        }
    }
    if (has_prompt("tarjan") || (has_prompt("strongly") && has_prompt("connected")) ||
        (has_prompt("connected") && has_prompt("component"))) {
        if (!(has_code("lowlink") && has_code("stack") && has_code("on_stack") &&
              has_code("components"))) {
            fail("missing_scc_invariants");
        }
    }
    if (has_prompt("lru") || (has_prompt("cache") && has_prompt("eviction"))) {
        if (!(has_code("def get") && has_code("def put") && has_code("order") &&
              has_code("pop(0)") && has_code("del cache"))) {
            fail("missing_lru_invariants");
        }
    }
    if (has_prompt("binary") && has_prompt("search")) {
        if (!(has_code("low") && has_code("high") && has_code("mid") &&
              has_code("while low <= high") && has_code("return -1"))) {
            fail("missing_binary_search_invariants");
        }
    }
    if (has_prompt("merge") && has_prompt("sort")) {
        if (!(has_code("values[:mid]") && has_code("values[mid:]") &&
              has_code("while i < len(left)") && has_code("result.extend"))) {
            fail("missing_merge_sort_invariants");
        }
    }
    if ((has_prompt("two") && has_prompt("sum")) ||
        (has_prompt("pair") && (has_prompt("index") || has_prompt("indices")))) {
        if (!(has_code("seen") && has_code("enumerate") &&
              has_code("complement = target - value") &&
              has_code("return [seen[complement], index]"))) {
            fail("missing_two_sum_invariants");
        }
    }
    if (has_prompt("sliding") && has_prompt("window")) {
        const bool explicit_window =
            has_code("range(0, len(values) - k + 1)") &&
            has_code("window = values[start:start + k]") &&
            has_code("result.append(max(window))");
        const bool monotonic_queue =
            has_code("deque") &&
            has_code("while queue") &&
            has_code("queue.popleft") &&
            has_code("values[queue[-1]]") &&
            has_code("result.append(values[queue[0]])");
        if (!(explicit_window || monotonic_queue)) {
            fail("missing_sliding_window_invariants");
        }
    }
    if (has_prompt("disjoint") || (has_prompt("union") && has_prompt("find"))) {
        if (!(has_code("parent") && has_code("rank") &&
              has_code("def find") && has_code("def union") &&
              has_code("parent[x] = find(parent[x])") &&
              has_code("return find, union, connected"))) {
            fail("missing_union_find_invariants");
        }
    }
    return report;
}

inline PythonCodeVerification verify_python_behavior_against_prompt(std::string_view prompt,
                                                                    std::string_view code) {
    PythonCodeVerification report;
    report.ok = true;
    const auto lower_prompt = code_verifier_lower(prompt);
    const auto lower_code = code_verifier_lower(code);
    const auto has_prompt = [&](std::string_view token) {
        return lower_prompt.find(token) != std::string::npos;
    };
    const auto has_code = [&](std::string_view token) {
        return lower_code.find(token) != std::string::npos;
    };
    const auto fail = [&](std::string issue) {
        report.ok = false;
        code_verifier_add_unique(report.issues, std::move(issue));
    };

    const bool two_sum_prompt =
        (has_prompt("two") && has_prompt("sum")) ||
        (has_prompt("pair") && (has_prompt("index") || has_prompt("indices")));
    const bool sum_sequence_prompt =
        !two_sum_prompt &&
        has_prompt("sum") &&
        (has_prompt("number") || has_prompt("numbers") ||
         has_prompt("list") || has_prompt("values") || has_prompt("sequence"));
    if (sum_sequence_prompt) {
        const bool accumulator_behavior =
            has_code("= 0") &&
            has_code("for ") &&
            has_code("+=") &&
            (has_code("return total") || has_code("return result"));
        if (!accumulator_behavior || has_code("return sum(")) {
            fail("missing_sum_accumulator_behavior");
        }
    }

    if (two_sum_prompt) {
        const bool complement_behavior =
            has_code("seen") &&
            has_code("for ") &&
            has_code("enumerate") &&
            has_code("target - value") &&
            has_code("return [seen[complement], index]");
        if (!complement_behavior) {
            fail("missing_two_sum_behavior");
        }
    }

    if (has_prompt("topological") || (has_prompt("dependency") && has_prompt("graph"))) {
        const bool kahn_behavior =
            has_code("indegree") &&
            has_code("ready") &&
            has_code("while ready") &&
            has_code("len(order)") &&
            has_code("raise valueerror");
        if (!kahn_behavior) {
            fail("missing_topological_cycle_behavior");
        }
    }

    if (has_prompt("sliding") && has_prompt("window")) {
        const bool monotonic_queue_behavior =
            has_code("deque") &&
            has_code("while queue") &&
            has_code("queue.popleft") &&
            has_code("queue.pop") &&
            has_code("result.append(values[queue[0]])");
        const bool direct_window_behavior =
            has_code("for start in range") &&
            has_code("values[start:start + k]") &&
            has_code("result.append(max(window))");
        if (!(monotonic_queue_behavior || direct_window_behavior)) {
            fail("missing_sliding_window_behavior");
        }
    }

    if (has_prompt("tarjan") || (has_prompt("strongly") && has_prompt("connected")) ||
        (has_prompt("connected") && has_prompt("component"))) {
        const bool tarjan_behavior =
            (has_code("lowlink") || has_code("lowlinks")) &&
            has_code("stack") &&
            has_code("on_stack") &&
            has_code("visit(") &&
            has_code("components.append");
        if (!tarjan_behavior) {
            fail("missing_scc_behavior");
        }
    }

    return report;
}

} // namespace dzeta
