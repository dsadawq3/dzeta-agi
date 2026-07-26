#include "token_field.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::string read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    assert(input);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

dzeta::OscillatorField trained_field(std::size_t threads) {
    const std::vector<std::string> corpus{
        "red apple grows sweet fruit beside a warm kitchen",
        "blue ocean waves carry a small boat home",
        "green forest moss hides a quiet path",
        "silver robot learns safe local tools",
    };
    dzeta::OscillatorField field(4096, 128, 777);
    field.set_generation_temperature(0.30L);
    field.set_thread_count(threads);
    field.set_parallel_min_dimensions(1);
    for (int pass = 0; pass < 3; ++pass) {
        for (const auto& line : corpus) {
            field.learn(line);
        }
    }
    return field;
}

}  // namespace

int main() {
    // Sampled generation (temperature > 0) must be reproducible from the
    // constructor seed, and bit-identical across thread counts.
    auto serial = trained_field(1);
    auto parallel = trained_field(4);

    const std::string first_serial = serial.forward("silver robot", 8);
    const std::string first_parallel = parallel.forward("silver robot", 8);
    assert(!first_serial.empty());
    assert(first_serial == first_parallel);

    // The seeded stream continues identically on the next call too.
    assert(serial.forward("red apple", 8) == parallel.forward("red apple", 8));

    // Saving the same state twice must produce byte-identical files
    // (long double padding is zeroed on write).
    const auto base = std::filesystem::temp_directory_path();
    const auto path_a = base / "dzeta_determinism_a.bin";
    const auto path_b = base / "dzeta_determinism_b.bin";
    serial.save_model(path_a.string());
    serial.save_model(path_b.string());
    assert(read_bytes(path_a) == read_bytes(path_b));
    std::filesystem::remove(path_a);
    std::filesystem::remove(path_b);

    std::cout << "dzeta_determinism passed\n";
    return 0;
}
