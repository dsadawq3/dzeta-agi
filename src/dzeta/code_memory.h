#pragma once

#include "adaptive_tokenizer.h"
#include "sat.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dzeta {

inline bool is_identifier_start(unsigned char ch) {
    return std::isalpha(ch) != 0 || ch == '_';
}

inline bool is_identifier_body(unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '_';
}

inline std::vector<std::string> tokenize_code(std::string_view text, std::size_t max_tokens = 0) {
    std::vector<std::string> tokens;
    std::size_t i = 0;
    while (i < text.size()) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (max_tokens != 0 && tokens.size() >= max_tokens) {
            break;
        }
        if (ch == '\r') {
            ++i;
            continue;
        }
        if (ch == '\n') {
            tokens.emplace_back("\n");
            ++i;
            continue;
        }
        if (std::isspace(ch) != 0) {
            ++i;
            continue;
        }
        if (is_identifier_start(ch)) {
            const std::size_t begin = i++;
            while (i < text.size() && is_identifier_body(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            tokens.emplace_back(text.substr(begin, i - begin));
            continue;
        }
        if (std::isdigit(ch) != 0) {
            const std::size_t begin = i++;
            while (i < text.size()) {
                const unsigned char next = static_cast<unsigned char>(text[i]);
                if (std::isalnum(next) == 0 && next != '_' && next != '.') {
                    break;
                }
                ++i;
            }
            tokens.emplace_back(text.substr(begin, i - begin));
            continue;
        }
        if (ch == '\'' || ch == '"') {
            const char quote = static_cast<char>(ch);
            ++i;
            bool escaped = false;
            while (i < text.size()) {
                const char current = text[i++];
                if (escaped) {
                    escaped = false;
                    continue;
                }
                if (current == '\\') {
                    escaped = true;
                    continue;
                }
                if (current == quote) {
                    break;
                }
            }
            tokens.emplace_back("\"...\"");
            continue;
        }

        if (i + 1 < text.size()) {
            const std::string two{text.substr(i, 2)};
            if (two == "==" || two == "!=" || two == "<=" || two == ">=" || two == "->" ||
                two == ":=" || two == "+=" || two == "-=" || two == "*=" || two == "/=" ||
                two == "//" || two == "**") {
                tokens.push_back(two);
                i += 2;
                continue;
            }
        }
        tokens.emplace_back(1, static_cast<char>(ch));
        ++i;
    }
    return tokens;
}

class CodeTokenMemory {
public:
    void observe(std::string_view text) {
        adaptive_tokenizer_.observe(text);
        const auto tokens = adaptive_tokenizer_.encode(text);
        observe_tokens(tokens);
    }

    void observe_tokens(const std::vector<std::string>& tokens) {
        if (tokens.empty()) {
            return;
        }
        for (const auto& token : tokens) {
            ++unigrams_[token];
            ++observations_;
        }
        dirty_ranked_ = true;
    }

    void replace_counts(const std::vector<std::pair<std::string, std::size_t>>& counts) {
        replace_state(counts, {});
    }

    void replace_state(const std::vector<std::pair<std::string, std::size_t>>& counts,
                       const std::vector<std::pair<std::string, std::size_t>>& subwords) {
        unigrams_.clear();
        observations_ = 0;
        for (const auto& [token, count] : counts) {
            if (count == 0) {
                continue;
            }
            unigrams_[token] += count;
            observations_ += count;
        }
        adaptive_tokenizer_.replace_state(counts, subwords);
        dirty_ranked_ = true;
    }

    std::size_t token_count() const noexcept {
        return unigrams_.size();
    }

    std::size_t observation_count() const noexcept {
        return observations_;
    }

    std::size_t frequency(std::string_view token) const {
        const auto found = unigrams_.find(std::string(token));
        return found == unigrams_.end() ? 0 : found->second;
    }

    std::vector<std::pair<std::string, std::size_t>> ranked_tokens(std::size_t limit) const {
        refresh_ranked();
        const auto take = std::min(limit, ranked_unigrams_.size());
        return {ranked_unigrams_.begin(), ranked_unigrams_.begin() + take};
    }

    std::vector<std::string> encode_text(std::string_view text) const {
        return adaptive_tokenizer_.encode(text);
    }

    std::size_t tokenizer_subword_count() const noexcept {
        return adaptive_tokenizer_.subword_count();
    }

    std::vector<std::pair<std::string, std::size_t>> tokenizer_subwords() const {
        return adaptive_tokenizer_.subwords();
    }

private:
    void refresh_ranked() const {
        if (!dirty_ranked_) {
            return;
        }
        ranked_unigrams_.assign(unigrams_.begin(), unigrams_.end());
        std::sort(ranked_unigrams_.begin(), ranked_unigrams_.end(), [](const auto& left, const auto& right) {
            if (left.second == right.second) {
                return left.first < right.first;
            }
            return left.second > right.second;
        });
        dirty_ranked_ = false;
    }

    std::unordered_map<std::string, std::size_t> unigrams_;
    std::size_t observations_ = 0;
    mutable bool dirty_ranked_ = true;
    mutable std::vector<std::pair<std::string, std::size_t>> ranked_unigrams_;
    AdaptiveTokenizer adaptive_tokenizer_;
};

inline char hex_digit(unsigned int value) {
    return static_cast<char>(value < 10 ? ('0' + value) : ('a' + (value - 10)));
}

inline std::string hex_encode(std::string_view text) {
    std::string encoded;
    encoded.reserve(text.size() * 2);
    for (unsigned char ch : text) {
        encoded.push_back(hex_digit((ch >> 4U) & 0x0FU));
        encoded.push_back(hex_digit(ch & 0x0FU));
    }
    return encoded;
}

inline unsigned int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') {
        return static_cast<unsigned int>(ch - '0');
    }
    if (ch >= 'a' && ch <= 'f') {
        return static_cast<unsigned int>(ch - 'a' + 10);
    }
    if (ch >= 'A' && ch <= 'F') {
        return static_cast<unsigned int>(ch - 'A' + 10);
    }
    throw std::runtime_error("invalid hex token memory byte");
}

inline std::string hex_decode(std::string_view encoded) {
    if (encoded.size() % 2 != 0) {
        throw std::runtime_error("odd-length hex token memory field");
    }
    std::string decoded;
    decoded.reserve(encoded.size() / 2);
    for (std::size_t i = 0; i < encoded.size(); i += 2) {
        const auto high = hex_value(encoded[i]);
        const auto low = hex_value(encoded[i + 1]);
        decoded.push_back(static_cast<char>((high << 4U) | low));
    }
    return decoded;
}

inline void save_token_memory(const CodeTokenMemory& memory, std::string_view path) {
    std::ofstream output(std::string(path), std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write token memory: " + std::string(path));
    }
    output << "DZETA_TOKEN_MEMORY_V2\t" << memory.observation_count() << "\n";
    for (const auto& [token, count] : memory.ranked_tokens(memory.token_count())) {
        output << "token\t" << hex_encode(token) << '\t' << count << '\n';
    }
    for (const auto& [subword, count] : memory.tokenizer_subwords()) {
        output << "subword\t" << hex_encode(subword) << '\t' << count << '\n';
    }
}

inline CodeTokenMemory load_token_memory(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read token memory: " + std::string(path));
    }

    std::string header;
    std::getline(input, header);
    const std::string prefix_v1 = "DZETA_TOKEN_MEMORY_V1\t";
    const std::string prefix_v2 = "DZETA_TOKEN_MEMORY_V2\t";
    const bool v1 = header.rfind(prefix_v1, 0) == 0;
    const bool v2 = header.rfind(prefix_v2, 0) == 0;
    if (!v1 && !v2) {
        throw std::runtime_error("invalid token memory header: " + std::string(path));
    }

    std::vector<std::pair<std::string, std::size_t>> counts;
    std::vector<std::pair<std::string, std::size_t>> subwords;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto tab = line.find('\t');
        if (tab == std::string::npos) {
            throw std::runtime_error("invalid token memory row: " + std::string(path));
        }
        if (v1) {
            counts.emplace_back(hex_decode(std::string_view(line).substr(0, tab)),
                                static_cast<std::size_t>(std::stoull(line.substr(tab + 1))));
            continue;
        }

        const auto second = line.find('\t', tab + 1);
        if (second == std::string::npos) {
            throw std::runtime_error("invalid token memory v2 row: " + std::string(path));
        }
        const auto row_type = line.substr(0, tab);
        auto decoded = hex_decode(std::string_view(line).substr(tab + 1, second - tab - 1));
        const auto count = static_cast<std::size_t>(std::stoull(line.substr(second + 1)));
        if (row_type == "token") {
            counts.emplace_back(std::move(decoded), count);
        } else if (row_type == "subword") {
            subwords.emplace_back(std::move(decoded), count);
        } else {
            throw std::runtime_error("unknown token memory row type: " + std::string(path));
        }
    }

    CodeTokenMemory memory;
    memory.replace_state(counts, subwords);
    return memory;
}

inline std::string format_code_tokens(const std::vector<std::string>& tokens) {
    std::ostringstream out;
    std::string previous;
    int indent = 0;
    bool at_line_start = true;

    const auto no_space_before = [](std::string_view token) {
        return token == ")" || token == "]" || token == "}" || token == "," || token == ":" ||
               token == "." || token == "\n";
    };
    const auto no_space_after = [](std::string_view token) {
        return token == "(" || token == "[" || token == "{" || token == "." || token == "\n";
    };

    for (const auto& token : tokens) {
        if (token == "\n") {
            out << '\n';
            at_line_start = true;
            previous = token;
            continue;
        }
        if (at_line_start) {
            if (token == "return" || token == "if" || token == "for" || token == "while") {
                indent = std::max(indent, 1);
            }
            for (int i = 0; i < indent; ++i) {
                out << "    ";
            }
            at_line_start = false;
        } else if (!previous.empty() && !no_space_before(token) && !no_space_after(previous)) {
            out << ' ';
        }
        out << token;
        previous = token;
    }
    return out.str();
}

} // namespace dzeta
