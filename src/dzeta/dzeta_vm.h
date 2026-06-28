#pragma once

#include "sat.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

enum class VmProgramKind {
    Unknown,
    Identity,
    Factorial,
    IsPrime,
    Square,
    Increment,
    ReverseString,
    DedupeString,
    InfiniteLoop,
};

enum class BytecodeOp {
    ParseIntList,
    ParseText,
    GcdList,
    SumList,
    ProductList,
    MaxList,
    MinList,
    SortIntList,
    IsPalindrome,
    CountVowels,
    StringLength,
    WordCount,
    Halt,
};

struct BytecodeInstruction {
    BytecodeOp op = BytecodeOp::Halt;
    long long argument = 0;
};

struct BytecodeProgram {
    std::string name;
    std::vector<BytecodeInstruction> instructions;
    std::vector<std::string> trace;
};

struct VmProgram {
    VmProgramKind kind = VmProgramKind::Unknown;
    std::string name;
    std::vector<std::string> trace;
};

struct VmRunResult {
    bool ok = false;
    bool timed_out = false;
    std::string output;
    std::size_t steps = 0;
    std::string error;
};

inline std::string vm_program_kind_name(VmProgramKind kind) {
    switch (kind) {
    case VmProgramKind::Identity:
        return "identity";
    case VmProgramKind::Factorial:
        return "factorial";
    case VmProgramKind::IsPrime:
        return "is_prime";
    case VmProgramKind::Square:
        return "square";
    case VmProgramKind::Increment:
        return "increment";
    case VmProgramKind::ReverseString:
        return "reverse_string";
    case VmProgramKind::DedupeString:
        return "dedupe_string";
    case VmProgramKind::InfiniteLoop:
        return "infinite_loop";
    case VmProgramKind::Unknown:
        return "unknown";
    }
    return "unknown";
}

inline VmProgramKind parse_vm_program_kind(std::string_view text) {
    if (text == "identity") {
        return VmProgramKind::Identity;
    }
    if (text == "factorial") {
        return VmProgramKind::Factorial;
    }
    if (text == "is_prime") {
        return VmProgramKind::IsPrime;
    }
    if (text == "square") {
        return VmProgramKind::Square;
    }
    if (text == "increment") {
        return VmProgramKind::Increment;
    }
    if (text == "reverse_string") {
        return VmProgramKind::ReverseString;
    }
    if (text == "dedupe_string") {
        return VmProgramKind::DedupeString;
    }
    if (text == "infinite_loop") {
        return VmProgramKind::InfiniteLoop;
    }
    return VmProgramKind::Unknown;
}

inline std::optional<long long> parse_vm_integer(std::string_view input) {
    std::string text(input);
    text.erase(std::remove_if(text.begin(), text.end(), [](unsigned char ch) {
                   return std::isspace(ch) != 0;
               }),
               text.end());
    if (text.empty()) {
        return std::nullopt;
    }
    std::size_t consumed = 0;
    try {
        const auto value = std::stoll(text, &consumed);
        if (consumed == text.size()) {
            return value;
        }
    } catch (...) {
        return std::nullopt;
    }
    return std::nullopt;
}

inline std::vector<long long> parse_vm_integer_list(std::string_view input) {
    std::string text(input);
    for (auto& ch : text) {
        if (ch == ';' || ch == '\n' || ch == '\t') {
            ch = ',';
        }
    }
    std::stringstream stream(text);
    std::string item;
    std::vector<long long> values;
    while (std::getline(stream, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
                       return std::isspace(ch) != 0;
                   }),
                   item.end());
        if (item.empty()) {
            continue;
        }
        std::size_t consumed = 0;
        try {
            const auto value = std::stoll(item, &consumed);
            if (consumed != item.size()) {
                return {};
            }
            values.push_back(value);
        } catch (...) {
            return {};
        }
    }
    return values;
}

inline long long vm_gcd(long long a, long long b) {
    a = std::llabs(a);
    b = std::llabs(b);
    while (b != 0) {
        const long long t = a % b;
        a = b;
        b = t;
    }
    return a;
}

inline std::string join_vm_integer_list(std::vector<long long> values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    return out.str();
}

inline std::string vm_bool_string(bool value) {
    return value ? "true" : "false";
}

class DzetaVM {
public:
    VmRunResult execute(const VmProgram& program, std::string_view input, std::size_t step_limit = 4096) const {
        VmRunResult result;
        result.steps = 0;
        if (step_limit == 0) {
            result.timed_out = true;
            result.error = "step limit exhausted";
            return result;
        }
        switch (program.kind) {
        case VmProgramKind::Identity:
            result.ok = true;
            result.steps = 1;
            result.output = std::string(input);
            return result;
        case VmProgramKind::ReverseString:
            result.ok = true;
            result.steps = input.size();
            if (result.steps > step_limit) {
                result.ok = false;
                result.timed_out = true;
                result.output.clear();
                result.error = "step limit exhausted";
                return result;
            }
            result.output.assign(input.rbegin(), input.rend());
            return result;
        case VmProgramKind::DedupeString: {
            result.steps = input.size();
            if (result.steps > step_limit) {
                result.timed_out = true;
                result.error = "step limit exhausted";
                return result;
            }
            std::string out;
            for (char ch : input) {
                if (out.find(ch) == std::string::npos) {
                    out.push_back(ch);
                }
            }
            result.ok = true;
            result.output = std::move(out);
            return result;
        }
        case VmProgramKind::Factorial: {
            const auto parsed = parse_vm_integer(input);
            if (!parsed || *parsed < 0 || *parsed > 20) {
                result.error = "invalid factorial input";
                return result;
            }
            long long value = 1;
            for (long long i = 2; i <= *parsed; ++i) {
                if (++result.steps > step_limit) {
                    result.timed_out = true;
                    result.error = "step limit exhausted";
                    return result;
                }
                value *= i;
            }
            result.ok = true;
            result.output = std::to_string(value);
            return result;
        }
        case VmProgramKind::IsPrime: {
            const auto parsed = parse_vm_integer(input);
            if (!parsed) {
                result.error = "invalid prime input";
                return result;
            }
            bool prime = *parsed >= 2;
            for (long long d = 2; prime && d * d <= *parsed; ++d) {
                if (++result.steps > step_limit) {
                    result.timed_out = true;
                    result.error = "step limit exhausted";
                    return result;
                }
                if (*parsed % d == 0) {
                    prime = false;
                }
            }
            result.ok = true;
            result.output = vm_bool_string(prime);
            return result;
        }
        case VmProgramKind::Square: {
            const auto parsed = parse_vm_integer(input);
            if (!parsed || std::abs(*parsed) > 3'000'000LL) {
                result.error = "invalid square input";
                return result;
            }
            result.ok = true;
            result.steps = 2;
            result.output = std::to_string((*parsed) * (*parsed));
            return result;
        }
        case VmProgramKind::Increment: {
            const auto parsed = parse_vm_integer(input);
            if (!parsed) {
                result.error = "invalid increment input";
                return result;
            }
            result.ok = true;
            result.steps = 1;
            result.output = std::to_string(*parsed + 1);
            return result;
        }
        case VmProgramKind::InfiniteLoop:
            result.steps = step_limit;
            result.timed_out = true;
            result.error = "step limit exhausted";
            return result;
        case VmProgramKind::Unknown:
            result.error = "unknown program";
            return result;
        }
        result.error = "unreachable program state";
        return result;
    }

    VmRunResult execute(const BytecodeProgram& program, std::string_view input, std::size_t step_limit = 4096) const {
        VmRunResult result;
        std::vector<long long> int_list;
        std::string text;
        for (const auto& instruction : program.instructions) {
            if (++result.steps > step_limit) {
                result.timed_out = true;
                result.error = "step limit exhausted";
                return result;
            }
            switch (instruction.op) {
            case BytecodeOp::ParseIntList:
                int_list = parse_vm_integer_list(input);
                if (int_list.empty()) {
                    result.error = "invalid integer list";
                    return result;
                }
                break;
            case BytecodeOp::ParseText:
                text = std::string(input);
                break;
            case BytecodeOp::GcdList: {
                if (int_list.empty()) {
                    result.error = "gcd requires integer list";
                    return result;
                }
                long long value = int_list.front();
                for (std::size_t i = 1; i < int_list.size(); ++i) {
                    value = vm_gcd(value, int_list[i]);
                    if (++result.steps > step_limit) {
                        result.timed_out = true;
                        result.error = "step limit exhausted";
                        return result;
                    }
                }
                result.output = std::to_string(value);
                break;
            }
            case BytecodeOp::SumList: {
                long long sum = 0;
                for (const auto value : int_list) {
                    sum += value;
                    if (++result.steps > step_limit) {
                        result.timed_out = true;
                        result.error = "step limit exhausted";
                        return result;
                    }
                }
                result.output = std::to_string(sum);
                break;
            }
            case BytecodeOp::ProductList: {
                long long product = 1;
                for (const auto value : int_list) {
                    product *= value;
                    if (++result.steps > step_limit) {
                        result.timed_out = true;
                        result.error = "step limit exhausted";
                        return result;
                    }
                }
                result.output = std::to_string(product);
                break;
            }
            case BytecodeOp::MaxList:
                if (int_list.empty()) {
                    result.error = "max requires integer list";
                    return result;
                }
                result.output = std::to_string(*std::max_element(int_list.begin(), int_list.end()));
                break;
            case BytecodeOp::MinList:
                if (int_list.empty()) {
                    result.error = "min requires integer list";
                    return result;
                }
                result.output = std::to_string(*std::min_element(int_list.begin(), int_list.end()));
                break;
            case BytecodeOp::SortIntList:
                std::sort(int_list.begin(), int_list.end());
                result.output = join_vm_integer_list(int_list);
                break;
            case BytecodeOp::IsPalindrome: {
                std::string normalized;
                for (const unsigned char ch : text) {
                    if (std::isalnum(ch) != 0) {
                        normalized.push_back(static_cast<char>(std::tolower(ch)));
                    }
                }
                const bool palindrome = std::equal(normalized.begin(), normalized.begin() + static_cast<std::ptrdiff_t>(normalized.size() / 2U), normalized.rbegin());
                result.output = vm_bool_string(palindrome);
                break;
            }
            case BytecodeOp::CountVowels: {
                std::size_t count = 0;
                for (const unsigned char ch : text) {
                    const char lower = static_cast<char>(std::tolower(ch));
                    if (lower == 'a' || lower == 'e' || lower == 'i' || lower == 'o' || lower == 'u') {
                        ++count;
                    }
                    if (++result.steps > step_limit) {
                        result.timed_out = true;
                        result.error = "step limit exhausted";
                        return result;
                    }
                }
                result.output = std::to_string(count);
                break;
            }
            case BytecodeOp::StringLength:
                result.output = std::to_string(text.size());
                break;
            case BytecodeOp::WordCount: {
                std::stringstream stream(text);
                std::string word;
                std::size_t count = 0;
                while (stream >> word) {
                    ++count;
                }
                result.output = std::to_string(count);
                break;
            }
            case BytecodeOp::Halt:
                result.ok = true;
                return result;
            }
        }
        result.ok = true;
        return result;
    }
};

inline VmProgram make_vm_program(VmProgramKind kind, std::string name = {}) {
    VmProgram program;
    program.kind = kind;
    program.name = name.empty() ? vm_program_kind_name(kind) : std::move(name);
    program.trace.push_back("kind=" + vm_program_kind_name(kind));
    return program;
}

inline VmProgram make_infinite_loop_program() {
    return make_vm_program(VmProgramKind::InfiniteLoop, "infinite_loop");
}

class ProgramSynthesizer {
public:
    std::optional<VmProgram> synthesize(const std::vector<std::pair<std::string, std::string>>& examples) const {
        if (examples.empty()) {
            return std::nullopt;
        }
        const std::vector<VmProgramKind> candidates{
            VmProgramKind::Factorial,
            VmProgramKind::IsPrime,
            VmProgramKind::Square,
            VmProgramKind::Increment,
            VmProgramKind::ReverseString,
            VmProgramKind::DedupeString,
            VmProgramKind::Identity,
        };
        DzetaVM vm;
        for (const auto kind : candidates) {
            auto program = make_vm_program(kind);
            bool all_match = true;
            for (const auto& [input, expected] : examples) {
                const auto run = vm.execute(program, input, 8192);
                if (!run.ok || normalize_output(run.output) != normalize_output(expected)) {
                    all_match = false;
                    break;
                }
            }
            if (all_match) {
                program.trace.push_back("verified_examples=" + std::to_string(examples.size()));
                return program;
            }
        }
        return std::nullopt;
    }

    std::optional<BytecodeProgram> synthesize_bytecode(
        const std::vector<std::pair<std::string, std::string>>& examples,
        const std::vector<std::pair<std::string, std::string>>& heldouts = {}) const {
        if (examples.empty()) {
            return std::nullopt;
        }
        const std::vector<BytecodeProgram> candidates{
            make_bytecode_program("gcd", {BytecodeOp::ParseIntList, BytecodeOp::GcdList, BytecodeOp::Halt}),
            make_bytecode_program("sum_list", {BytecodeOp::ParseIntList, BytecodeOp::SumList, BytecodeOp::Halt}),
            make_bytecode_program("product_list", {BytecodeOp::ParseIntList, BytecodeOp::ProductList, BytecodeOp::Halt}),
            make_bytecode_program("max_list", {BytecodeOp::ParseIntList, BytecodeOp::MaxList, BytecodeOp::Halt}),
            make_bytecode_program("min_list", {BytecodeOp::ParseIntList, BytecodeOp::MinList, BytecodeOp::Halt}),
            make_bytecode_program("sort_small_list", {BytecodeOp::ParseIntList, BytecodeOp::SortIntList, BytecodeOp::Halt}),
            make_bytecode_program("palindrome", {BytecodeOp::ParseText, BytecodeOp::IsPalindrome, BytecodeOp::Halt}),
            make_bytecode_program("count_vowels", {BytecodeOp::ParseText, BytecodeOp::CountVowels, BytecodeOp::Halt}),
            make_bytecode_program("string_length", {BytecodeOp::ParseText, BytecodeOp::StringLength, BytecodeOp::Halt}),
            make_bytecode_program("word_count", {BytecodeOp::ParseText, BytecodeOp::WordCount, BytecodeOp::Halt}),
        };
        DzetaVM vm;
        for (auto candidate : candidates) {
            if (verifies(vm, candidate, examples) && verifies(vm, candidate, heldouts)) {
                candidate.trace.push_back("cegis_examples=" + std::to_string(examples.size()));
                candidate.trace.push_back("heldouts=" + std::to_string(heldouts.size()));
                candidate.trace.push_back("compression=" + std::to_string(candidate.instructions.size()));
                return candidate;
            }
        }
        return std::nullopt;
    }

private:
    static BytecodeProgram make_bytecode_program(std::string name, std::vector<BytecodeOp> ops) {
        BytecodeProgram program;
        program.name = std::move(name);
        for (const auto op : ops) {
            program.instructions.push_back({op, 0});
        }
        program.trace.push_back("bytecode_template=" + program.name);
        return program;
    }

    static bool verifies(const DzetaVM& vm,
                         const BytecodeProgram& program,
                         const std::vector<std::pair<std::string, std::string>>& examples) {
        for (const auto& [input, expected] : examples) {
            const auto run = vm.execute(program, input, 8192);
            if (!run.ok || normalize_output(run.output) != normalize_output(expected)) {
                return false;
            }
        }
        return true;
    }

    static std::string normalize_output(std::string_view text) {
        std::string out(text);
        out.erase(std::remove_if(out.begin(), out.end(), [](unsigned char ch) {
                      return std::isspace(ch) != 0;
                  }),
                  out.end());
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return out;
    }
};

struct SkillRecord {
    std::string name;
    VmProgram program;
    long double confidence = 0.0L;
};

class SkillGenome {
public:
    void store(std::string name, VmProgram program, long double confidence) {
        for (auto& record : records_) {
            if (record.name == name) {
                record.program = std::move(program);
                record.confidence = std::clamp(confidence, 0.0L, 1.0L);
                return;
            }
        }
        records_.push_back({std::move(name), std::move(program), std::clamp(confidence, 0.0L, 1.0L)});
    }

    std::optional<VmProgram> find(std::string_view name) const {
        for (const auto& record : records_) {
            if (record.name == name || record.program.name == name) {
                return record.program;
            }
        }
        return std::nullopt;
    }

    const std::vector<SkillRecord>& records() const noexcept {
        return records_;
    }

private:
    std::vector<SkillRecord> records_;

    friend void save_skill_genome(const SkillGenome&, std::string_view);
    friend SkillGenome load_skill_genome(std::string_view);
};

inline void save_skill_genome(const SkillGenome& genome, std::string_view path) {
    std::ofstream output(std::string(path), std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write skill genome: " + std::string(path));
    }
    output << "DZETA_SKILL_GENOME_V1\n";
    for (const auto& record : genome.records_) {
        output << record.name << '\t'
               << vm_program_kind_name(record.program.kind) << '\t'
               << record.program.name << '\t'
               << static_cast<double>(record.confidence) << '\n';
    }
}

inline SkillGenome load_skill_genome(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read skill genome: " + std::string(path));
    }
    std::string header;
    std::getline(input, header);
    if (header != "DZETA_SKILL_GENOME_V1") {
        throw std::runtime_error("invalid skill genome: " + std::string(path));
    }
    SkillGenome genome;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream stream(line);
        std::string name;
        std::string kind;
        std::string program_name;
        std::string confidence;
        std::getline(stream, name, '\t');
        std::getline(stream, kind, '\t');
        std::getline(stream, program_name, '\t');
        std::getline(stream, confidence, '\t');
        auto program = make_vm_program(parse_vm_program_kind(kind), program_name);
        genome.store(name, std::move(program), confidence.empty() ? 0.0L : std::stold(confidence));
    }
    return genome;
}

} // namespace dzeta
