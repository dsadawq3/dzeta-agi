#pragma once

#include "cpu_accel.h"
#include "zeta_rhythm.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

namespace dzeta {

struct ZetaCacheStats {
    std::size_t theta_hits = 0;
    std::size_t theta_misses = 0;
    std::size_t energy_hits = 0;
    std::size_t energy_misses = 0;
};

class ZetaCache {
public:
    long double theta(std::uint32_t prime) {
        const auto found = theta_cache_.find(prime);
        if (found != theta_cache_.end()) {
            ++stats_.theta_hits;
            return found->second;
        }
        ++stats_.theta_misses;
        const auto value = riemann_siegel_theta(prime);
        theta_cache_[prime] = value;
        return value;
    }

    long double spectral_energy_cached(std::uint32_t prime, std::size_t zeros) {
        const auto key = std::make_pair(prime, zeros);
        const auto found = energy_cache_.find(key);
        if (found != energy_cache_.end()) {
            ++stats_.energy_hits;
            return found->second;
        }
        ++stats_.energy_misses;
        const auto value = spectral_energy(prime, zeros);
        energy_cache_[key] = value;
        return value;
    }

    const ZetaCacheStats& stats() const noexcept {
        return stats_;
    }

private:
    std::map<std::uint32_t, long double> theta_cache_;
    std::map<std::pair<std::uint32_t, std::size_t>, long double> energy_cache_;
    ZetaCacheStats stats_;
};

struct PerformanceReport {
    long double training_mib_per_second = 0.0L;
    std::size_t zeta_cache_hits = 0;
    std::size_t candidate_verifications = 0;
    long double field_eval_ms = 0.0L;
    long double memory_retrieval_ms = 0.0L;
    long double vm_synthesis_ms = 0.0L;
};

struct PerformanceCounters {
    std::uintmax_t training_bytes = 0;
    long double training_seconds = 0.0L;
    std::size_t zeta_cache_hits = 0;
    std::size_t candidate_verifications = 0;
    long double field_eval_ms = 0.0L;
    long double memory_retrieval_ms = 0.0L;
    long double vm_synthesis_ms = 0.0L;

    void add_training_bytes(std::uintmax_t bytes, long double seconds) {
        training_bytes += bytes;
        training_seconds += std::max(0.0L, seconds);
    }

    PerformanceReport report() const {
        PerformanceReport out;
        if (training_seconds > 1.0e-12L) {
            out.training_mib_per_second = static_cast<long double>(training_bytes) /
                                          (1024.0L * 1024.0L) / training_seconds;
        }
        out.zeta_cache_hits = zeta_cache_hits;
        out.candidate_verifications = candidate_verifications;
        out.field_eval_ms = field_eval_ms;
        out.memory_retrieval_ms = memory_retrieval_ms;
        out.vm_synthesis_ms = vm_synthesis_ms;
        return out;
    }
};

inline std::string performance_report_line(const PerformanceReport& report) {
    return "profile training_mib_s=" + std::to_string(static_cast<double>(report.training_mib_per_second)) +
           " zeta_cache_hits=" + std::to_string(report.zeta_cache_hits) +
           " candidate_verifications=" + std::to_string(report.candidate_verifications);
}

} // namespace dzeta
