#include "token_field.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

int main() {
    const std::string text = "the little robot started walking";
    const auto tokens = dzeta::tokenize_code(text);

    assert(tokens.size() > 5);
    assert(std::any_of(tokens.begin(), tokens.end(), [](const std::string& token) {
        return dzeta::is_subword_continuation(token);
    }));
    assert(dzeta::format_code_tokens(tokens) == text);

    dzeta::OscillatorField field(4096, 128, 12345);
    field.set_generation_temperature(0.0L);
    for (int i = 0; i < 3; ++i) {
        field.learn("the little robot started walking into a beautiful garden");
        field.learn("the little robot started looking at special treasures");
    }
    const auto output = field.forward("the little robot", 8);
    assert(output.find("##") == std::string::npos);

    std::cout << "dzeta_tokenizer passed\n";
    return 0;
}
