#pragma once

#include <string_view>

namespace dzeta {

enum class KernerPhase {
    FreeFlow,
    SynchronizedFlow,
    WideMovingJam,
};

inline std::string_view phase_name(KernerPhase phase) {
    switch (phase) {
    case KernerPhase::FreeFlow:
        return "F/free-flow";
    case KernerPhase::SynchronizedFlow:
        return "S/synchronized-flow";
    case KernerPhase::WideMovingJam:
        return "J/wide-moving-jam";
    }
    return "unknown";
}

inline KernerPhase transition_phase(long double coherent_density,
                                    long double threshold_f_to_s,
                                    long double threshold_s_to_j) {
    if (coherent_density >= threshold_s_to_j) {
        return KernerPhase::WideMovingJam;
    }
    if (coherent_density >= threshold_f_to_s) {
        return KernerPhase::SynchronizedFlow;
    }
    return KernerPhase::FreeFlow;
}

} // namespace dzeta
