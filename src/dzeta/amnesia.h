#pragma once

#include "cloud.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace dzeta {

struct AmnesiaMask {
    std::vector<std::size_t> blocked_indices;
};

inline AmnesiaMask cut_amnesia_window(Cloud& cloud, std::size_t begin, std::size_t count) {
    AmnesiaMask mask;
    auto& handles = cloud.handles();
    if (begin >= handles.size() || count == 0) {
        return mask;
    }

    const std::size_t end = std::min(handles.size(), begin + count);
    mask.blocked_indices.reserve(end - begin);
    for (std::size_t index = begin; index < end; ++index) {
        handles[index].blocked = true;
        handles[index].activation = 0.0L;
        mask.blocked_indices.push_back(index);
    }
    return mask;
}

inline void restore_amnesia_window(Cloud& cloud, const AmnesiaMask& mask) {
    auto& handles = cloud.handles();
    for (const auto index : mask.blocked_indices) {
        if (index < handles.size()) {
            handles[index].blocked = false;
        }
    }
}

inline void controlled_death(Cloud& cloud, const AmnesiaMask& mask) {
    auto& handles = cloud.handles();
    const auto max_resistance = cloud.config().max_resistance;
    for (const auto index : mask.blocked_indices) {
        if (index < handles.size()) {
            handles[index].state = 0.0L;
            handles[index].activation = 0.0L;
            handles[index].resistance = max_resistance;
            handles[index].blocked = true;
        }
    }
}

} // namespace dzeta
