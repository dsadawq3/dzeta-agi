#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct WorkingMemorySlot {
    std::string content;
    long double salience = 0.0L;
    std::size_t age = 0;
    bool refreshed = false;
};

class WorkingMemory {
public:
    explicit WorkingMemory(std::size_t capacity = 7)
        : capacity_(std::clamp<std::size_t>(capacity, 5, 9)) {}

    std::size_t remember(std::string content, long double salience) {
        if (slots_.size() >= capacity_) {
            const auto victim = std::min_element(slots_.begin(), slots_.end(), [](const auto& left, const auto& right) {
                const auto left_score = left.salience - 0.03L * static_cast<long double>(left.age);
                const auto right_score = right.salience - 0.03L * static_cast<long double>(right.age);
                return left_score < right_score;
            });
            slots_.erase(victim);
        }
        slots_.push_back({std::move(content), std::clamp(salience, 0.0L, 1.0L), 0, false});
        return slots_.size() - 1U;
    }

    void refresh(std::size_t index) {
        if (index >= slots_.size()) {
            return;
        }
        slots_[index].age = 0;
        slots_[index].salience = std::min(1.0L, slots_[index].salience + 0.20L);
        slots_[index].refreshed = true;
    }

    void decay(std::size_t ticks) {
        for (auto& slot : slots_) {
            slot.age += ticks;
            if (!slot.refreshed) {
                slot.salience *= std::pow(0.97L, static_cast<long double>(ticks));
            }
            slot.refreshed = false;
        }
    }

    bool contains(std::string_view content) const {
        return std::any_of(slots_.begin(), slots_.end(), [&](const auto& slot) {
            return slot.content == content;
        });
    }

    const std::vector<WorkingMemorySlot>& slots() const noexcept {
        return slots_;
    }

private:
    std::size_t capacity_ = 7;
    std::vector<WorkingMemorySlot> slots_;
};

} // namespace dzeta
