#include "token_field.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::vector<std::string> corpus{
        "red apple grows sweet fruit beside a warm kitchen",
        "blue ocean waves carry a small boat home",
        "green forest moss hides a quiet path",
        "silver robot learns safe local tools",
    };

    dzeta::OscillatorField serial(4096, 128, 12345);
    serial.set_generation_temperature(0.0L);
    serial.set_thread_count(1);
    serial.set_parallel_min_dimensions(1);

    dzeta::OscillatorField parallel(4096, 128, 12345);
    parallel.set_generation_temperature(0.0L);
    parallel.set_thread_count(4);
    parallel.set_parallel_min_dimensions(1);

    for (int pass = 0; pass < 4; ++pass) {
        for (const auto& line : corpus) {
            serial.learn(line);
            parallel.learn(line);
        }
    }

    assert(serial.size() == parallel.size());
    assert(serial.observation_count() == parallel.observation_count());
    assert(serial.contrastive_update_count() == parallel.contrastive_update_count());
    assert(std::isfinite(static_cast<double>(parallel.mean_loss())));
    assert(parallel.thread_count() == 4);
    assert(serial.forward("red apple", 6) == parallel.forward("red apple", 6));
    assert(serial.forward("silver robot", 6) == parallel.forward("silver robot", 6));

    std::cout << "dzeta_parallel passed\n";
    return 0;
}
