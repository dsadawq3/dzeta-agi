#pragma once

#include <algorithm>

namespace dzeta {

inline long double entropy_singularity_score(long double normalized_entropy,
                                             long double clique_pressure,
                                             long double mutual_information_norm,
                                             long double bridge_score) {
    const long double noise = std::clamp(normalized_entropy, 0.0L, 1.0L);
    const long double no_structure = 1.0L - std::clamp(clique_pressure, 0.0L, 1.0L);
    const long double no_information = 1.0L - std::clamp(mutual_information_norm, 0.0L, 1.0L);
    const long double bridge_failure = 1.0L - std::clamp(bridge_score, 0.0L, 1.0L);
    return std::clamp(0.35L * noise + 0.30L * no_structure +
                          0.20L * no_information + 0.15L * bridge_failure,
                      0.0L,
                      1.0L);
}

} // namespace dzeta
