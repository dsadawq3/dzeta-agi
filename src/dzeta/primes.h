#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dzeta {

inline bool is_prime(std::uint64_t value) {
    if (value < 2) {
        return false;
    }
    if (value == 2 || value == 3) {
        return true;
    }
    if (value % 2 == 0 || value % 3 == 0) {
        return false;
    }
    for (std::uint64_t divisor = 5; divisor * divisor <= value; divisor += 6) {
        if (value % divisor == 0 || value % (divisor + 2) == 0) {
            return false;
        }
    }
    return true;
}

inline std::size_t nth_prime_upper_bound(std::size_t count) {
    if (count < 6) {
        return 16;
    }
    const long double n = static_cast<long double>(count);
    const long double estimate = n * (std::log(n) + std::log(std::log(n)));
    return static_cast<std::size_t>(estimate + 32.0L);
}

inline std::vector<std::uint32_t> generate_first_primes(std::size_t count) {
    if (count == 0) {
        return {};
    }

    std::size_t limit = nth_prime_upper_bound(count);
    for (;;) {
        std::vector<bool> composite(limit + 1, false);
        for (std::size_t p = 2; p * p <= limit; ++p) {
            if (!composite[p]) {
                for (std::size_t multiple = p * p; multiple <= limit; multiple += p) {
                    composite[multiple] = true;
                }
            }
        }

        std::vector<std::uint32_t> primes;
        primes.reserve(count);
        for (std::size_t value = 2; value <= limit && primes.size() < count; ++value) {
            if (!composite[value]) {
                primes.push_back(static_cast<std::uint32_t>(value));
            }
        }

        if (primes.size() == count) {
            return primes;
        }
        limit *= 2;
    }
}

inline std::uint32_t twin_prime_of(std::uint32_t prime) {
    const std::uint64_t candidate = static_cast<std::uint64_t>(prime) + 2ULL;
    if (candidate <= UINT32_MAX && is_prime(candidate)) {
        return static_cast<std::uint32_t>(candidate);
    }
    return 0;
}

inline bool has_twin_prime(std::uint32_t prime) {
    return twin_prime_of(prime) != 0;
}

} // namespace dzeta
