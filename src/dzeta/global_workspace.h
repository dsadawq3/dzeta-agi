#pragma once

#include "mentalese_core.h"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dzeta {

struct WorkspaceItem {
    std::string source;
    MentalState state;
    long double salience = 0.0L;
};

struct WorkspaceBroadcast {
    std::vector<WorkspaceItem> items;
    MentalState self_state;
};

class GlobalWorkspace {
public:
    void publish(std::string source, MentalState state, long double salience) {
        items_.push_back({std::move(source), std::move(state), std::clamp(salience, 0.0L, 1.0L)});
    }

    WorkspaceBroadcast broadcast(std::size_t limit) const {
        WorkspaceBroadcast out;
        out.items = items_;
        std::sort(out.items.begin(), out.items.end(), [](const auto& left, const auto& right) {
            return left.salience > right.salience;
        });
        if (out.items.size() > limit) {
            out.items.resize(limit);
        }
        out.self_state = merge_items(out.items);
        mental_add_symbol(out.self_state, "workspace_broadcast");
        return out;
    }

    std::string introspect() const {
        const auto top = broadcast(3);
        std::ostringstream out;
        out << "workspace items=" << top.items.size()
            << " uncertainty=" << static_cast<double>(top.self_state.uncertainty);
        for (const auto& item : top.items) {
            out << " source=" << item.source;
        }
        return out.str();
    }

private:
    static MentalState merge_items(const std::vector<WorkspaceItem>& items) {
        MentalState merged;
        if (items.empty()) {
            merged.uncertainty = 0.95L;
            return merged;
        }
        const std::size_t dimension = items.front().state.vectors.size();
        merged.vectors.assign(dimension, 0.0L);
        long double salience_sum = 0.0L;
        long double uncertainty_sum = 0.0L;
        for (const auto& item : items) {
            salience_sum += item.salience;
            uncertainty_sum += item.salience * item.state.uncertainty;
            for (std::size_t i = 0; i < std::min(dimension, item.state.vectors.size()); ++i) {
                merged.vectors[i] += item.salience * item.state.vectors[i];
            }
            for (const auto& symbol : item.state.symbols) {
                mental_add_symbol(merged, symbol);
            }
            for (const auto& goal : item.state.goals) {
                merged.goals.push_back(goal);
            }
        }
        if (salience_sum > 1.0e-12L) {
            for (auto& value : merged.vectors) {
                value /= salience_sum;
            }
            merged.uncertainty = std::clamp(uncertainty_sum / salience_sum, 0.02L, 0.95L);
        }
        mental_add_symbol(merged, "self_state");
        merged.trace_id = stable_hash(introspection_seed(merged));
        return merged;
    }

    static std::string introspection_seed(const MentalState& state) {
        std::string seed;
        for (const auto& symbol : state.symbols) {
            seed += symbol;
            seed.push_back('|');
        }
        return seed;
    }

    std::vector<WorkspaceItem> items_;
};

} // namespace dzeta
