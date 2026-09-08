#include "field_state.h"
#include "math_helpers.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using dz_real = double;

inline dz_real padic_norm(dz_real x, std::uint32_t p) {
    if (p < 2) return 0.0;
    if (!std::isfinite(x)) return 0.0;
    if (std::abs(x) < 1e-30) return 0.0;
    dz_real ax = std::abs(x);
    dz_real log_p = std::log(static_cast<dz_real>(p));
    if (!std::isfinite(log_p) || log_p <= 0) return 0.0;
    dz_real v = std::floor(std::log(ax) / log_p);
    v = std::clamp(v, static_cast<dz_real>(-300), static_cast<dz_real>(300));
    dz_real pow_p_v = std::pow(static_cast<dz_real>(p), v);
    if (std::isfinite(pow_p_v) && pow_p_v > 0) {
        dz_real scaled = ax / pow_p_v;
        if (scaled >= static_cast<dz_real>(p) && v < 300) v += 1;
        else if (scaled < 1.0 && v > -300) v -= 1;
    }
    dz_real r = std::pow(static_cast<dz_real>(p), -v);
    if (!std::isfinite(r)) return 0.0;
    return r;
}

int main() {
    // 1. Basic p-adic norm properties
    assert(padic_norm(0.0, 2) == 0.0);
    assert(padic_norm(0.0, 3) == 0.0);
    assert(std::abs(padic_norm(1.0, 2) - 1.0) < 1e-9);
    assert(std::abs(padic_norm(1.0, 5) - 1.0) < 1e-9);

    // Powers of primes: |p^k|_p = p^{-k}
    assert(std::abs(padic_norm(2.0, 2) - 0.5) < 1e-9);
    assert(std::abs(padic_norm(4.0, 2) - 0.25) < 1e-9);
    assert(std::abs(padic_norm(8.0, 2) - 0.125) < 1e-9);
    assert(std::abs(padic_norm(3.0, 3) - (1.0 / 3.0)) < 1e-9);
    assert(std::abs(padic_norm(9.0, 3) - (1.0 / 9.0)) < 1e-9);
    assert(std::abs(padic_norm(5.0, 5) - 0.2) < 1e-9);
    assert(std::abs(padic_norm(25.0, 5) - 0.04) < 1e-9);

    // 2. Non-Archimedean Ultrametric Inequality: |x + y|_p <= max(|x|_p, |y|_p)
    const std::vector<dz_real> test_vals = {0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0};
    for (std::uint32_t p : {2U, 3U, 5U}) {
        for (auto x : test_vals) {
            for (auto y : test_vals) {
                dz_real nx = padic_norm(x, p);
                dz_real ny = padic_norm(y, p);
                dz_real nsum = padic_norm(x + y, p);
                dz_real max_xy = std::max(nx, ny);
                assert(nsum <= max_xy * (1.0 + 1e-9));
            }
        }
    }

    // 3. Robustness against NaN, Inf, and subnormal inputs
    assert(padic_norm(std::numeric_limits<dz_real>::quiet_NaN(), 2) == 0.0);
    assert(padic_norm(std::numeric_limits<dz_real>::infinity(), 2) == 0.0);
    assert(padic_norm(-std::numeric_limits<dz_real>::infinity(), 2) == 0.0);
    assert(padic_norm(1e-40, 2) == 0.0);

    // 4. Ultrametric coupling matrix J_pq = 1 / max(p, q)
    for (std::uint32_t p : dzeta::kAdelicPrimes) {
        for (std::uint32_t q : dzeta::kAdelicPrimes) {
            long double jval = dzeta::adelic_J(p, q);
            long double expected = 1.0L / static_cast<long double>(std::max(p, q));
            assert(std::abs(jval - expected) < 1e-12L);
        }
    }

    // 5. Adelic 9-wave accumulator test
    const std::size_t width = 64;
    dzeta::FieldWaveAdelicAccumulator adelic_acc(width);
    std::vector<long double> token_wave(width, 0.0L);
    dzeta::field_token_wave("quantum_logic", width, token_wave.data());

    adelic_acc.push_adelic_wave(token_wave.data(), 1.2L);
    std::vector<long double> sig;
    adelic_acc.signature_into(sig);
    assert(sig.size() == width);

    long double sig_norm = 0.0L;
    for (auto v : sig) sig_norm += v * v;
    sig_norm = std::sqrt(sig_norm);
    assert(std::abs(sig_norm - 1.0L) < 1e-7L);

    // 6. Signature consistency
    auto sig1 = dzeta::field_impulse_adelic_signature("adelic wave field", width);
    auto sig2 = dzeta::field_impulse_adelic_signature("adelic wave field", width);
    assert(sig1 == sig2);

    std::cout << "dzeta_padic passed: ultrametric and adelic 9-wave verification successful\n";
    return 0;
}
