#pragma once

#include "mentalese_core.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct SelfModuleSpec {
    std::string name;
    std::string source;
};

class ModuleRegistry {
public:
    std::size_t compile_and_register(const SelfModuleSpec& spec) {
        auto program = compile_source(spec);
        if (!program) {
            throw std::runtime_error("invalid self module source: " + spec.name);
        }
        ThoughtModule module;
        module.name = spec.name;
        module.input_signature = "mentalese:any";
        module.output_signature = "bytecode:" + program->name;
        module.program = *program;
        module.trainable_weights.assign(8, 0.25L);
        module.enabled = true;
        modules_.push_back(std::move(module));
        return modules_.size();
    }

    std::optional<ThoughtModule> find(std::string_view name) const {
        for (const auto& module : modules_) {
            if (module.name == name && module.enabled) {
                return module;
            }
        }
        return std::nullopt;
    }

    const std::vector<ThoughtModule>& modules() const noexcept {
        return modules_;
    }

private:
    static BytecodeProgram make_program(std::string name, std::vector<BytecodeOp> ops) {
        BytecodeProgram program;
        program.name = std::move(name);
        for (const auto op : ops) {
            program.instructions.push_back({op, 0});
        }
        program.trace.push_back("self_module_compiled");
        return program;
    }

    static std::optional<BytecodeProgram> compile_source(const SelfModuleSpec& spec) {
        std::string source = spec.source;
        std::transform(source.begin(), source.end(), source.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (source.find("count_vowels") != std::string::npos) {
            return make_program("count_vowels", {BytecodeOp::ParseText, BytecodeOp::CountVowels, BytecodeOp::Halt});
        }
        if (source.find("gcd") != std::string::npos) {
            return make_program("gcd", {BytecodeOp::ParseIntList, BytecodeOp::GcdList, BytecodeOp::Halt});
        }
        if (source.find("palindrome") != std::string::npos) {
            return make_program("palindrome", {BytecodeOp::ParseText, BytecodeOp::IsPalindrome, BytecodeOp::Halt});
        }
        if (source.find("sum_list") != std::string::npos) {
            return make_program("sum_list", {BytecodeOp::ParseIntList, BytecodeOp::SumList, BytecodeOp::Halt});
        }
        if (source.find("product_list") != std::string::npos) {
            return make_program("product_list", {BytecodeOp::ParseIntList, BytecodeOp::ProductList, BytecodeOp::Halt});
        }
        if (source.find("max_list") != std::string::npos) {
            return make_program("max_list", {BytecodeOp::ParseIntList, BytecodeOp::MaxList, BytecodeOp::Halt});
        }
        if (source.find("min_list") != std::string::npos) {
            return make_program("min_list", {BytecodeOp::ParseIntList, BytecodeOp::MinList, BytecodeOp::Halt});
        }
        if (source.find("word_count") != std::string::npos) {
            return make_program("word_count", {BytecodeOp::ParseText, BytecodeOp::WordCount, BytecodeOp::Halt});
        }
        if (source.find("sort") != std::string::npos) {
            return make_program("sort_small_list", {BytecodeOp::ParseIntList, BytecodeOp::SortIntList, BytecodeOp::Halt});
        }
        return std::nullopt;
    }

    std::vector<ThoughtModule> modules_;
};

} // namespace dzeta
