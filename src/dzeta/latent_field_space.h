#pragma once

#include "field_state.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct LatentVector {
    std::vector<long double> values;
};

inline void normalize_latent(LatentVector& vector) {
    const long double norm = std::sqrt(std::inner_product(vector.values.begin(),
                                                          vector.values.end(),
                                                          vector.values.begin(),
                                                          0.0L));
    if (norm <= 1.0e-18L) {
        return;
    }
    for (auto& value : vector.values) {
        value /= norm;
    }
}

inline long double latent_cosine(const LatentVector& left, const LatentVector& right) {
    return field_cosine_similarity(left.values, right.values);
}

class LatentFieldSpace {
public:
    explicit LatentFieldSpace(std::size_t dimension = 32, std::uint64_t seed = 0xD237AULL)
        : dimension_(std::max<std::size_t>(1, dimension)), seed_(seed) {}

    std::size_t dimension() const noexcept {
        return dimension_;
    }

    LatentVector encode(std::string_view text) const {
        LatentVector vector;
        vector.values.assign(dimension_, 0.0L);
        const auto tokens = tokenize_query(text);
        for (std::size_t token_index = 0; token_index < tokens.size(); ++token_index) {
            std::uint64_t state = stable_hash(tokens[token_index]) ^
                                  (seed_ + 0x9e3779b97f4a7c15ULL * (token_index + 1U));
            for (std::size_t dim = 0; dim < dimension_; ++dim) {
                const long double wave = 2.0L * field_unit_from_hash(splitmix64(state)) - 1.0L;
                const long double position = std::sin(static_cast<long double>((token_index + 1U) * (dim + 1U)) * 0.173L);
                vector.values[dim] += wave * (1.0L + 0.15L * position);
            }
        }
        normalize_latent(vector);
        return vector;
    }

    LatentVector encode_field(const FieldState& field) const {
        LatentVector vector;
        vector.values.assign(dimension_, 0.0L);
        if (field.empty()) {
            return vector;
        }
        for (std::size_t i = 0; i < field.size(); ++i) {
            const std::size_t dim = static_cast<std::size_t>(field.primes[i] % dimension_);
            vector.values[dim] += std::cos(field.phases[i]) * (0.25L + field.activations[i]) +
                                  0.25L * field.semantic_charge[i];
            vector.values[(dim + 7U) % dimension_] += std::sin(field.theta[i]) *
                                                       (0.1L + field.activations[i]);
        }
        normalize_latent(vector);
        return vector;
    }

    LatentVector blend(const LatentVector& left, const LatentVector& right, long double alpha) const {
        alpha = std::clamp(alpha, 0.0L, 1.0L);
        LatentVector out;
        out.values.assign(dimension_, 0.0L);
        for (std::size_t i = 0; i < dimension_; ++i) {
            const long double a = i < left.values.size() ? left.values[i] : 0.0L;
            const long double b = i < right.values.size() ? right.values[i] : 0.0L;
            out.values[i] = (1.0L - alpha) * a + alpha * b;
        }
        normalize_latent(out);
        return out;
    }

private:
    std::size_t dimension_;
    std::uint64_t seed_;
};

} // namespace dzeta
