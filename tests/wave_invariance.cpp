#include "field_state.h"

#include <cassert>
#include <iostream>
#include <string>

// Pure-kernel invariance properties of the multi-scale context waves.
// Thresholds carry margin below/above the values measured at tuning time
// (shift 0.94, order 0.75, horizon 1.00, insertion 0.98, unrelated 0.20).
int main() {
    const std::size_t width = 64;
    const auto sig = [&](const std::string& text) {
        return dzeta::field_impulse_signature(text, width);
    };

    // (a) Shift transfer: the same recent context at a different absolute
    // offset must stay strongly correlated. The old absolute-position waves
    // scored ~0.0 here by construction.
    const auto shift = dzeta::field_cosine_similarity(
        sig("silver comet streaks across night"),
        sig("yesterday we watched the silver comet streaks across night"));
    assert(shift > 0.85L);

    // (b) Order sensitivity: a reversed window must be measurably farther
    // than a shifted copy of the same window.
    const auto order = dzeta::field_cosine_similarity(
        sig("alpha beta gamma delta"),
        sig("delta gamma beta alpha"));
    assert(order < shift - 0.10L);

    // (c) Horizon decay: context older than the slow horizon must stop
    // influencing the signature.
    std::string filler;
    for (int i = 0; i < 60; ++i) {
        filler += "filler" + std::to_string(i) + " ";
    }
    const auto horizon = dzeta::field_cosine_similarity(
        sig("apple banana cherry " + filler + "quiet fox jumps over rivers"),
        sig("table window chair " + filler + "quiet fox jumps over rivers"));
    assert(horizon > 0.95L);

    // (d) One-token insertion degrades gracefully, not catastrophically.
    const auto insertion = dzeta::field_cosine_similarity(
        sig("the quiet fox jumps over rivers"),
        sig("the very quiet fox jumps over rivers"));
    assert(insertion > 0.90L);

    // (e) Unrelated text stays well separated.
    const auto unrelated = dzeta::field_cosine_similarity(
        sig("silver comet streaks across night"),
        sig("historian compares archives treaties maps"));
    assert(unrelated < 0.35L);

    // (f) The kernel is a pure function: bitwise deterministic.
    assert(sig("the little robot started walking") == sig("the little robot started walking"));

    std::cout << "dzeta_wave_invariance passed: shift=" << static_cast<double>(shift)
              << " order=" << static_cast<double>(order)
              << " horizon=" << static_cast<double>(horizon) << "\n";
    return 0;
}
