#pragma once

#include <cstdint>

namespace dzeta {

struct Handle {
    std::uint32_t prime = 2;
    std::uint32_t twin_prime = 0;
    long double state = 0.0L;
    long double resistance = 0.5L;
    long double activation = 0.0L;
    long double energy = 0.0L;
    long double theta = 0.0L;
    long double phase = 0.0L;
    bool blocked = false;
    bool tuned = false;
};

} // namespace dzeta
