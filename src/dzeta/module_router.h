#pragma once

#include "mentalese_core.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct RoutedModule {
    ThoughtModule module;
    long double score = 0.0L;
};

class ModuleRouter {
public:
    void register_module(ThoughtModule module) {
        if (module.trainable_weights.empty()) {
            module.trainable_weights.assign(8, 0.25L);
        }
        weights_[module.name] = std::max<long double>(0.15L, average(module.trainable_weights));
        modules_.push_back(std::move(module));
    }

    std::vector<RoutedModule> route(const MentalState& state, std::size_t top_k) const {
        std::vector<RoutedModule> routed;
        for (const auto& module : modules_) {
            if (!module.enabled) {
                continue;
            }
            const long double signature_score = state.has_symbol(module.input_signature) ? 0.45L : 0.0L;
            const long double lexical_score = symbol_overlap(state, module.name);
            const auto found = weights_.find(module.name);
            const long double learned = found == weights_.end() ? 0.25L : found->second;
            const long double score = std::clamp(0.42L * learned + signature_score + 0.22L * lexical_score,
                                                 0.0L, 1.0L);
            routed.push_back({module, score});
        }
        std::sort(routed.begin(), routed.end(), [](const auto& left, const auto& right) {
            return left.score > right.score;
        });
        if (routed.size() > top_k) {
            routed.resize(top_k);
        }
        return routed;
    }

    void apply_feedback(std::string_view module_name, long double reward) {
        auto& weight = weights_[std::string(module_name)];
        if (weight <= 0.0L) {
            weight = 0.25L;
        }
        weight = std::clamp(weight + 0.12L * std::clamp(reward, -1.0L, 1.0L), 0.01L, 1.0L);
    }

private:
    static long double average(const std::vector<long double>& values) {
        if (values.empty()) {
            return 0.0L;
        }
        long double sum = 0.0L;
        for (const auto value : values) {
            sum += value;
        }
        return sum / static_cast<long double>(values.size());
    }

    static long double symbol_overlap(const MentalState& state, std::string_view module_name) {
        long double score = 0.0L;
        for (const auto& symbol : state.symbols) {
            if (module_name.find(symbol) != std::string_view::npos ||
                symbol.find(module_name) != std::string::npos) {
                score += 0.2L;
            }
        }
        return std::min(1.0L, score);
    }

    std::vector<ThoughtModule> modules_;
    std::map<std::string, long double> weights_;
};

} // namespace dzeta
