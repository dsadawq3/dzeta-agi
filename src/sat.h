#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct SatClause {
    std::vector<int> literals;
};

struct SatProblem {
    std::size_t variable_count = 0;
    std::vector<SatClause> clauses;
};

struct ComplexitySpectrum {
    std::vector<std::size_t> walk_times;
    long double mean_unsatisfied = 0.0L;
    long double temperature_hint = 0.05L;
};

inline std::uint64_t stable_hash(std::string_view text) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

inline std::vector<std::string> tokenize_query(std::string_view query) {
    std::vector<std::string> tokens;
    std::string current;
    for (const unsigned char ch : query) {
        if (std::isalnum(ch) != 0) {
            current.push_back(static_cast<char>(std::tolower(ch)));
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    if (tokens.empty()) {
        tokens.emplace_back("empty");
    }
    return tokens;
}

inline SatProblem sat_from_query(std::string_view query) {
    const auto tokens = tokenize_query(query);
    SatProblem problem;
    problem.variable_count = std::clamp<std::size_t>(tokens.size() * 2, 1, 64);

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        std::uint64_t seed = stable_hash(tokens[i]) ^ (i * 0x9e3779b97f4a7c15ULL);
        SatClause clause;
        const std::size_t literal_count = 2 + (splitmix64(seed) % 2);
        clause.literals.reserve(literal_count);
        for (std::size_t j = 0; j < literal_count; ++j) {
            const auto raw = splitmix64(seed);
            const int variable = static_cast<int>((raw % problem.variable_count) + 1);
            const bool negated = ((raw >> 9U) & 1U) != 0;
            clause.literals.push_back(negated ? -variable : variable);
        }
        problem.clauses.push_back(std::move(clause));
    }

    for (std::size_t i = 1; i < tokens.size(); ++i) {
        SatClause bridge;
        const int left = static_cast<int>((stable_hash(tokens[i - 1]) % problem.variable_count) + 1);
        const int right = static_cast<int>((stable_hash(tokens[i]) % problem.variable_count) + 1);
        bridge.literals = {left, -right};
        problem.clauses.push_back(std::move(bridge));
    }

    return problem;
}

inline bool clause_satisfied(const SatClause& clause, const std::vector<bool>& assignment) {
    for (const int literal : clause.literals) {
        const auto variable = static_cast<std::size_t>(std::abs(literal) - 1);
        const bool value = variable < assignment.size() && assignment[variable];
        if ((literal > 0 && value) || (literal < 0 && !value)) {
            return true;
        }
    }
    return false;
}

inline std::size_t unsatisfied_count(const SatProblem& problem, const std::vector<bool>& assignment) {
    std::size_t count = 0;
    for (const auto& clause : problem.clauses) {
        if (!clause_satisfied(clause, assignment)) {
            ++count;
        }
    }
    return count;
}

inline ComplexitySpectrum complexity_spectrum(const SatProblem& problem,
                                              std::size_t walkers,
                                              std::size_t max_steps,
                                              std::uint64_t seed) {
    ComplexitySpectrum spectrum;
    spectrum.walk_times.reserve(walkers);
    std::mt19937_64 rng(seed);

    long double total_unsatisfied = 0.0L;
    for (std::size_t walker = 0; walker < walkers; ++walker) {
        std::vector<bool> assignment(problem.variable_count);
        for (auto bit = assignment.begin(); bit != assignment.end(); ++bit) {
            *bit = (rng() & 1U) != 0;
        }

        std::size_t best_unsat = unsatisfied_count(problem, assignment);
        std::size_t elapsed = 0;
        for (; elapsed < max_steps && best_unsat != 0; ++elapsed) {
            const auto variable = static_cast<std::size_t>(rng() % problem.variable_count);
            assignment[variable] = !assignment[variable];
            const auto candidate_unsat = unsatisfied_count(problem, assignment);
            if (candidate_unsat <= best_unsat || (rng() % 7U) == 0U) {
                best_unsat = candidate_unsat;
            } else {
                assignment[variable] = !assignment[variable];
            }
        }
        spectrum.walk_times.push_back(elapsed);
        total_unsatisfied += static_cast<long double>(best_unsat);
    }

    spectrum.mean_unsatisfied = walkers == 0 ? 0.0L : total_unsatisfied / static_cast<long double>(walkers);
    const long double clause_scale = problem.clauses.empty()
                                         ? 0.0L
                                         : spectrum.mean_unsatisfied / static_cast<long double>(problem.clauses.size());
    spectrum.temperature_hint = 0.02L + 0.30L * clause_scale +
                                0.002L * static_cast<long double>(problem.variable_count);
    return spectrum;
}

} // namespace dzeta
