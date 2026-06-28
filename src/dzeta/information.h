#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dzeta {

struct KolmogorovEstimate {
    std::size_t lz_phrases = 0;
    long double byte_entropy_bits = 0.0L;
    long double block_entropy_bits = 0.0L;
    long double compression_ratio_upper_bound = 0.0L;
    long double normalized_complexity = 0.0L;
};

inline long double shannon_entropy_bits(const std::vector<std::size_t>& counts) {
    std::size_t total = 0;
    for (const auto count : counts) {
        total += count;
    }
    if (total == 0) {
        return 0.0L;
    }
    long double entropy = 0.0L;
    for (const auto count : counts) {
        if (count == 0) {
            continue;
        }
        const long double p = static_cast<long double>(count) / static_cast<long double>(total);
        entropy -= p * std::log2(p);
    }
    return entropy;
}

inline long double normalized_entropy_bits(const std::vector<std::size_t>& counts) {
    std::size_t nonzero = 0;
    for (const auto count : counts) {
        if (count != 0) {
            ++nonzero;
        }
    }
    if (nonzero <= 1) {
        return 0.0L;
    }
    return shannon_entropy_bits(counts) / std::log2(static_cast<long double>(nonzero));
}

inline long double mutual_information_bits(const std::vector<std::size_t>& left,
                                           const std::vector<std::size_t>& right) {
    const std::size_t n = std::min(left.size(), right.size());
    if (n == 0) {
        return 0.0L;
    }

    std::unordered_map<std::size_t, std::size_t> left_counts;
    std::unordered_map<std::size_t, std::size_t> right_counts;
    std::unordered_map<std::uint64_t, std::size_t> joint_counts;
    for (std::size_t i = 0; i < n; ++i) {
        ++left_counts[left[i]];
        ++right_counts[right[i]];
        const std::uint64_t key = (static_cast<std::uint64_t>(left[i]) << 32U) ^
                                  static_cast<std::uint64_t>(right[i]);
        ++joint_counts[key];
    }

    long double mi = 0.0L;
    for (const auto& [key, joint] : joint_counts) {
        const auto lx = static_cast<std::size_t>(key >> 32U);
        const auto ry = static_cast<std::size_t>(key & 0xffffffffULL);
        const long double pxy = static_cast<long double>(joint) / static_cast<long double>(n);
        const long double px = static_cast<long double>(left_counts[lx]) / static_cast<long double>(n);
        const long double py = static_cast<long double>(right_counts[ry]) / static_cast<long double>(n);
        mi += pxy * std::log2(pxy / (px * py));
    }
    return std::max(0.0L, mi);
}

inline long double joint_entropy_bits(const std::vector<std::size_t>& left,
                                      const std::vector<std::size_t>& right) {
    const std::size_t n = std::min(left.size(), right.size());
    if (n == 0) {
        return 0.0L;
    }
    std::unordered_map<std::uint64_t, std::size_t> joint_counts;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t key = (static_cast<std::uint64_t>(left[i]) << 32U) ^
                                  static_cast<std::uint64_t>(right[i]);
        ++joint_counts[key];
    }
    std::vector<std::size_t> counts;
    counts.reserve(joint_counts.size());
    for (const auto& item : joint_counts) {
        counts.push_back(item.second);
    }
    return shannon_entropy_bits(counts);
}

inline long double normalized_mutual_information(const std::vector<std::size_t>& left,
                                                 const std::vector<std::size_t>& right) {
    if (left.empty() || right.empty()) {
        return 0.0L;
    }
    const auto left_max = *std::max_element(left.begin(), left.end());
    const auto right_max = *std::max_element(right.begin(), right.end());
    std::vector<std::size_t> left_counts(left_max + 1U, 0);
    std::vector<std::size_t> right_counts(right_max + 1U, 0);
    for (auto value : left) {
        ++left_counts[value];
    }
    for (auto value : right) {
        ++right_counts[value];
    }
    const long double hx = shannon_entropy_bits(left_counts);
    const long double hy = shannon_entropy_bits(right_counts);
    const long double denom = std::sqrt(std::max(0.0L, hx * hy));
    if (denom <= 0.0L) {
        return 0.0L;
    }
    return std::clamp(mutual_information_bits(left, right) / denom, 0.0L, 1.0L);
}

inline long double variation_of_information_bits(const std::vector<std::size_t>& left,
                                                 const std::vector<std::size_t>& right) {
    if (left.empty() || right.empty()) {
        return 0.0L;
    }
    const auto left_max = *std::max_element(left.begin(), left.end());
    const auto right_max = *std::max_element(right.begin(), right.end());
    std::vector<std::size_t> left_counts(left_max + 1U, 0);
    std::vector<std::size_t> right_counts(right_max + 1U, 0);
    for (auto value : left) {
        ++left_counts[value];
    }
    for (auto value : right) {
        ++right_counts[value];
    }
    const long double hx = shannon_entropy_bits(left_counts);
    const long double hy = shannon_entropy_bits(right_counts);
    return std::max(0.0L, hx + hy - 2.0L * mutual_information_bits(left, right));
}

inline long double conditional_entropy_bits(const std::vector<std::size_t>& left,
                                            const std::vector<std::size_t>& given) {
    std::vector<std::size_t> left_counts;
    std::vector<std::size_t> given_counts;
    const auto left_max = left.empty() ? 0 : *std::max_element(left.begin(), left.end());
    const auto given_max = given.empty() ? 0 : *std::max_element(given.begin(), given.end());
    left_counts.resize(left_max + 1U, 0);
    given_counts.resize(given_max + 1U, 0);
    for (auto value : left) {
        ++left_counts[value];
    }
    for (auto value : given) {
        ++given_counts[value];
    }
    return std::max(0.0L, shannon_entropy_bits(left_counts) - mutual_information_bits(left, given));
}

inline std::size_t lz78_phrase_count(std::string_view text) {
    std::unordered_set<std::string> dictionary;
    std::string current;
    std::size_t phrases = 0;
    for (char ch : text) {
        current.push_back(ch);
        if (dictionary.find(current) == dictionary.end()) {
            dictionary.insert(current);
            current.clear();
            ++phrases;
        }
    }
    if (!current.empty()) {
        ++phrases;
    }
    return phrases;
}

inline long double byte_entropy(std::string_view text) {
    std::vector<std::size_t> counts(256, 0);
    for (unsigned char ch : text) {
        ++counts[ch];
    }
    return shannon_entropy_bits(counts);
}

inline long double block_entropy(std::string_view text) {
    if (text.size() < 2) {
        return byte_entropy(text);
    }
    std::unordered_map<std::uint16_t, std::size_t> counts;
    for (std::size_t i = 0; i + 1 < text.size(); ++i) {
        const auto key = static_cast<std::uint16_t>(
            (static_cast<unsigned int>(static_cast<unsigned char>(text[i])) << 8U) |
            static_cast<unsigned int>(static_cast<unsigned char>(text[i + 1])));
        ++counts[key];
    }
    std::vector<std::size_t> values;
    values.reserve(counts.size());
    for (const auto& item : counts) {
        values.push_back(item.second);
    }
    return shannon_entropy_bits(values);
}

inline KolmogorovEstimate kolmogorov_upper_bound(std::string_view text) {
    KolmogorovEstimate estimate;
    if (text.empty()) {
        return estimate;
    }
    estimate.lz_phrases = lz78_phrase_count(text);
    estimate.byte_entropy_bits = byte_entropy(text);
    estimate.block_entropy_bits = block_entropy(text);
    const long double n = static_cast<long double>(text.size());
    const long double lz_norm = static_cast<long double>(estimate.lz_phrases) / n;
    const long double entropy_norm = std::clamp(estimate.byte_entropy_bits / 8.0L, 0.0L, 1.0L);
    const long double block_norm = std::clamp(estimate.block_entropy_bits / 16.0L, 0.0L, 1.0L);
    estimate.compression_ratio_upper_bound = std::clamp(
        (static_cast<long double>(estimate.lz_phrases) * std::max(1.0L, std::log2(n + 1.0L))) /
            (8.0L * n),
        0.0L,
        1.0L);
    estimate.normalized_complexity = std::clamp(0.50L * lz_norm + 0.35L * entropy_norm + 0.15L * block_norm,
                                                0.0L,
                                                1.0L);
    return estimate;
}

inline std::vector<std::size_t> histogram_bins(const std::vector<long double>& values,
                                               long double min_value,
                                               long double max_value,
                                               std::size_t bin_count) {
    std::vector<std::size_t> counts(std::max<std::size_t>(1, bin_count), 0);
    if (values.empty() || max_value <= min_value) {
        return counts;
    }
    const long double scale = static_cast<long double>(counts.size()) / (max_value - min_value);
    for (auto value : values) {
        const auto bin = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::max(0.0L, (value - min_value) * scale)),
            0,
            counts.size() - 1U);
        ++counts[bin];
    }
    return counts;
}

} // namespace dzeta
