#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

inline std::vector<std::string> tokenize_code(std::string_view text, std::size_t max_tokens = 0) {
    std::vector<std::string> tokens;
    std::string current;
    for (std::size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (max_tokens > 0 && tokens.size() >= max_tokens) break;
        if (c == ' ' || c == '	' || c == '
' || c == '') {
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
            if (c == '
') tokens.push_back("\n");
            continue;
        }
        if (c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == ':' || c == ',' || c == ';' || c == '.') {
            if (!current.empty()) { tokens.push_back(current); current.clear(); }
            tokens.push_back(std::string(1, c));
            continue;
        }
        current += c;
    }
    if (!current.empty() && (max_tokens == 0 || tokens.size() < max_tokens))
        tokens.push_back(current);
    return tokens;
}

} // namespace dzeta
