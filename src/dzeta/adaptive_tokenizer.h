#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dzeta {

inline bool adaptive_is_word_char(unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '_';
}

class AdaptiveTokenizer {
public:
    void observe(std::string_view text) {
        for (const auto& token : scan_atoms(text)) {
            ++vocabulary_[token];
            if (is_word(token)) {
                learn_subwords(token, 1);
            }
        }
        dirty_ = true;
    }

    std::vector<std::string> encode(std::string_view text) const {
        std::vector<std::string> encoded;
        for (const auto& atom : scan_atoms(text)) {
            if (!is_word(atom) || frequency(atom) != 0) {
                encoded.push_back(atom);
                continue;
            }
            const auto pieces = split_unknown_word(atom);
            encoded.insert(encoded.end(), pieces.begin(), pieces.end());
        }
        return encoded;
    }

    std::size_t frequency(std::string_view token) const {
        const auto found = vocabulary_.find(std::string(token));
        return found == vocabulary_.end() ? 0 : found->second;
    }

    std::vector<std::pair<std::string, std::size_t>> vocabulary() const {
        refresh_ranked();
        return ranked_;
    }

    std::size_t subword_count() const noexcept {
        return subwords_.size();
    }

    std::size_t subword_frequency(std::string_view token) const {
        const auto found = subwords_.find(std::string(token));
        return found == subwords_.end() ? 0 : found->second;
    }

    std::vector<std::pair<std::string, std::size_t>> subwords() const {
        std::vector<std::pair<std::string, std::size_t>> ranked(subwords_.begin(), subwords_.end());
        std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
            if (left.second == right.second) {
                return left.first < right.first;
            }
            return left.second > right.second;
        });
        return ranked;
    }

    void replace_state(const std::vector<std::pair<std::string, std::size_t>>& vocabulary,
                       const std::vector<std::pair<std::string, std::size_t>>& subwords) {
        vocabulary_.clear();
        subwords_.clear();
        for (const auto& [token, count] : vocabulary) {
            if (count == 0) {
                continue;
            }
            vocabulary_[token] += count;
        }
        for (const auto& [subword, count] : subwords) {
            if (count == 0) {
                continue;
            }
            subwords_[subword] += count;
        }
        if (subwords_.empty()) {
            for (const auto& [token, count] : vocabulary_) {
                if (is_word(token)) {
                    learn_subwords(token, count);
                }
            }
        }
        dirty_ = true;
    }

private:
    static std::vector<std::string> scan_atoms(std::string_view text) {
        std::vector<std::string> atoms;
        std::size_t i = 0;
        while (i < text.size()) {
            const unsigned char ch = static_cast<unsigned char>(text[i]);
            if (ch == '\r') {
                ++i;
                continue;
            }
            if (ch == '\n') {
                atoms.emplace_back("\n");
                ++i;
                continue;
            }
            if (std::isspace(ch) != 0) {
                ++i;
                continue;
            }
            if (adaptive_is_word_char(ch)) {
                const std::size_t begin = i++;
                while (i < text.size() && adaptive_is_word_char(static_cast<unsigned char>(text[i]))) {
                    ++i;
                }
                atoms.emplace_back(text.substr(begin, i - begin));
                continue;
            }
            if (i + 1 < text.size()) {
                const std::string two{text.substr(i, 2)};
                if (two == "==" || two == "!=" || two == "<=" || two == ">=" || two == "->" ||
                    two == ":=" || two == "+=" || two == "-=" || two == "*=" || two == "/=" ||
                    two == "//" || two == "**") {
                    atoms.push_back(two);
                    i += 2;
                    continue;
                }
            }
            atoms.emplace_back(1, static_cast<char>(ch));
            ++i;
        }
        return atoms;
    }

    static bool is_word(const std::string& token) {
        return !token.empty() && std::all_of(token.begin(), token.end(), [](char ch) {
            return adaptive_is_word_char(static_cast<unsigned char>(ch));
        });
    }

    void learn_subwords(const std::string& word, std::size_t weight) {
        if (word.size() < 4) {
            return;
        }
        const std::size_t max_len = std::min<std::size_t>(10, word.size());
        for (std::size_t len = 3; len <= max_len; ++len) {
            for (std::size_t begin = 0; begin + len <= word.size(); ++begin) {
                subwords_[word.substr(begin, len)] += weight;
            }
        }
    }

    std::vector<std::string> split_unknown_word(const std::string& word) const {
        std::vector<std::string> pieces;
        std::size_t i = 0;
        while (i < word.size()) {
            std::string best;
            const std::size_t max_len = std::min<std::size_t>(10, word.size() - i);
            for (std::size_t len = max_len; len >= 2; --len) {
                const std::string candidate = word.substr(i, len);
                if (subwords_.find(candidate) != subwords_.end()) {
                    best = candidate;
                    break;
                }
                if (len == 2) {
                    break;
                }
            }
            if (best.empty()) {
                pieces.emplace_back(1, word[i]);
                ++i;
            } else {
                pieces.push_back(best);
                i += best.size();
            }
        }
        return pieces;
    }

    void refresh_ranked() const {
        if (!dirty_) {
            return;
        }
        ranked_.assign(vocabulary_.begin(), vocabulary_.end());
        std::sort(ranked_.begin(), ranked_.end(), [](const auto& left, const auto& right) {
            if (left.second == right.second) {
                return left.first < right.first;
            }
            return left.second > right.second;
        });
        dirty_ = false;
    }

    std::unordered_map<std::string, std::size_t> vocabulary_;
    std::unordered_map<std::string, std::size_t> subwords_;
    mutable bool dirty_ = true;
    mutable std::vector<std::pair<std::string, std::size_t>> ranked_;
};

} // namespace dzeta
