#include "token_field.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

int main() {
    dzeta::OscillatorField field(4096, 64, 12345);
    field.set_generation_temperature(0.0L);

    field.learn("alpha beta gamma delta");
    const auto observations_after_first = field.observation_count();
    const auto loss_after_first = field.mean_loss();

    for (int i = 0; i < 8; ++i) {
        field.learn("alpha beta gamma delta");
    }

    assert(field.size() >= 4);
    assert(field.observation_count() > observations_after_first);
    assert(std::isfinite(static_cast<double>(field.mean_loss())));
    assert(field.mean_loss() <= loss_after_first + 1.0e-9L);

    const std::string deterministic = field.forward("alpha", 4);
    assert(!deterministic.empty());

    dzeta::OscillatorField stochastic(4096, 64, 0);
    stochastic.set_generation_temperature(0.25L);
    stochastic.learn("safe open intelligence helps people");
    stochastic.learn("safe open intelligence runs locally");
    assert(stochastic.mean_loss() >= 0.0L);

    dzeta::OscillatorField contrastive(4096, 64, 12345);
    contrastive.set_generation_temperature(0.0L);
    for (int i = 0; i < 10; ++i) {
        contrastive.learn("red apple grows sweet fruit");
        contrastive.learn("blue ocean waves deep water");
        contrastive.learn("green forest grows quiet moss");
    }
    assert(contrastive.contrastive_update_count() > 0);
    assert(contrastive.forward("red apple", 4) != contrastive.forward("blue ocean", 4));

    std::cout << "dzeta_learning passed: " << deterministic << "\n";
    return 0;
}
