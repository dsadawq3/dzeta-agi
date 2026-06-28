#pragma once

#include <string_view>

namespace dzeta {

struct DisabledKummerFilter {
    bool enabled = false;

    bool accepts(std::string_view) const noexcept {
        return true;
    }
};

} // namespace dzeta
