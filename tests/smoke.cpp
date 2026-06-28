#include "token_field.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    dzeta::OscillatorField field(1024, 64);
    field.learn("def hello name return name");
    field.learn("safe local intelligence should be inspectable");

    const std::string generated = field.forward("def", 4);
    assert(field.size() > 0);
    assert(!generated.empty());

    std::cout << "dzeta_smoke passed: " << generated << "\n";
    return 0;
}
