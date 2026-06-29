#include "token_field.h"

#include <cassert>
#include <iostream>

int main() {
    dzeta::OscillatorField field(4096, 128, 0);
    field.set_update_probability(0.0L);
    field.set_update_noise(0.01L);
    field.set_random_init_scale(0.01L);

    field.learn("silver robot learns safe local tools beside a quiet child");

    assert(field.size() > 0);
    assert(field.observation_count() == 0);
    assert(field.contrastive_update_count() == 0);

    field.set_update_probability(1.0L);
    field.learn("silver robot learns safe local tools beside a quiet child");

    assert(field.observation_count() > 0);

    std::cout << "dzeta_stochastic passed\n";
    return 0;
}
