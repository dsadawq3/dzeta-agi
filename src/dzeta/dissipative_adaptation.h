#pragma once

#include <algorithm>
#include <cmath>

namespace dzeta {

inline long double dissipative_adaptation_score(long double previous_entropy,
                                                long double current_entropy,
                                                long double absorbed_drive,
                                                long double stability_gain) {
    const long double entropy_production = std::max(0.0L, current_entropy - previous_entropy);
    const long double drive = std::log1p(std::max(0.0L, absorbed_drive));
    const long double stability = std::max(0.0L, stability_gain);
    const long double raw = 0.40L * std::tanh(entropy_production) +
                            0.30L * std::tanh(drive) +
                            0.30L * std::tanh(stability);
    return std::clamp(raw, 0.0L, 1.0L);
}

} // namespace dzeta
