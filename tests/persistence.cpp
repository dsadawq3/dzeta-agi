#include "token_field.h"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

int main() {
    dzeta::OscillatorField field(4096, 128, 12345);
    field.set_generation_temperature(0.0L);
    field.set_dimension_interference(0.02L);
    field.set_thread_count(2);
    field.set_parallel_min_dimensions(1);

    for (int i = 0; i < 4; ++i) {
        field.learn("red apple grows sweet fruit beside a warm kitchen");
        field.learn("blue ocean waves carry a small boat home");
        field.learn("silver robot learns safe local tools");
    }

    const auto before_size = field.size();
    const auto before_observations = field.observation_count();
    const std::string before = field.forward("red apple", 6);
    assert(!before.empty());

    const auto path = std::filesystem::temp_directory_path() / "dzeta_persistence_test.bin";
    field.save_model(path.string());

    dzeta::OscillatorField loaded(128, 16, 999);
    loaded.load_model(path.string());
    loaded.set_generation_temperature(0.0L);
    loaded.set_thread_count(2);
    loaded.set_parallel_min_dimensions(1);

    assert(loaded.size() == before_size);
    assert(loaded.observation_count() == before_observations);
    assert(loaded.dimension_interference() > 0.019L);
    assert(loaded.forward("red apple", 6) == before);

    std::filesystem::remove(path);
    std::cout << "dzeta_persistence passed\n";
    return 0;
}
