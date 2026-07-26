#include "token_field.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// Opt-in compact (v3, int16 max-abs quantized) persistence: much smaller
// files, auto-detected on load, greedy generation preserved.
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

    const std::string before = field.forward("red apple", 6);
    assert(!before.empty());

    const auto base = std::filesystem::temp_directory_path();
    const auto path_v2 = base / "dzeta_persistence_v3_full.bin";
    const auto path_v3 = base / "dzeta_persistence_v3_compact.bin";
    const auto path_v3b = base / "dzeta_persistence_v3_requant.bin";
    field.save_model(path_v2.string());
    field.save_model(path_v3.string(), /*compact=*/true);

    // Quantized vectors are int16 instead of 16-byte long doubles.
    assert(std::filesystem::file_size(path_v3) * 3 < std::filesystem::file_size(path_v2));

    dzeta::OscillatorField loaded(128, 16, 999);
    loaded.load_model(path_v3.string());
    loaded.set_generation_temperature(0.0L);
    loaded.set_thread_count(2);
    loaded.set_parallel_min_dimensions(1);
    assert(loaded.size() == field.size());
    assert(loaded.observation_count() == field.observation_count());
    const std::string after = loaded.forward("red apple", 6);
    if (after != before) {
        std::cout << "before=" << before << "\nafter=" << after << "\n";
    }
    assert(after == before);

    // Requantization is stable: compact save of an already-quantized field
    // reproduces the same greedy output.
    loaded.save_model(path_v3b.string(), /*compact=*/true);
    dzeta::OscillatorField requantized(128, 16, 998);
    requantized.load_model(path_v3b.string());
    requantized.set_generation_temperature(0.0L);
    requantized.set_thread_count(2);
    requantized.set_parallel_min_dimensions(1);
    assert(requantized.forward("red apple", 6) == before);

    // Corrupting the scheme byte must be rejected loudly. Offset: uint64
    // magic length + 15 magic bytes + uint32 version.
    {
        std::fstream corrupt(path_v3, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(8 + 15 + 4);
        const char bad = static_cast<char>(0xFF);
        corrupt.write(&bad, 1);
    }
    bool threw = false;
    try {
        dzeta::OscillatorField victim(128, 16, 997);
        victim.load_model(path_v3.string());
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);

    std::filesystem::remove(path_v2);
    std::filesystem::remove(path_v3);
    std::filesystem::remove(path_v3b);
    std::cout << "dzeta_persistence_v3 passed\n";
    return 0;
}
