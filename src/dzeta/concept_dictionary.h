#pragma once

#include "mentalese_core.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace dzeta {

struct ConceptAtom {
    std::string name;
    std::vector<long double> centroid;
    std::vector<std::string> symbols;
    std::vector<std::string> evidence_snippets;
    std::size_t observations = 0;
    long double mdl_gain = 0.0L;
};

struct ConceptHit {
    ConceptAtom atom;
    long double score = 0.0L;
};

class ConceptDictionary {
public:
    explicit ConceptDictionary(std::size_t max_atoms = 512)
        : max_atoms_(std::max<std::size_t>(16, max_atoms)) {}

    void observe(const MentalTrace& trace) {
        observe_state("trace_initial", trace.initial_state, trace.reward, trace.free_energy);
        observe_state("trace_final", trace.final_state, trace.reward, trace.free_energy);
        for (const auto& module : trace.modules_used) {
            auto module_state = encode_prompt_mental_state("module " + module);
            mental_add_symbol(module_state, "module:" + module);
            observe_state("module:" + module, module_state, trace.reward, trace.free_energy);
        }
    }

    std::size_t atom_count() const noexcept {
        return atoms_.size();
    }

    const std::vector<ConceptAtom>& atoms() const noexcept {
        return atoms_;
    }

    void restore_atom(ConceptAtom atom) {
        if (atom.centroid.empty() || atoms_.size() >= max_atoms_) {
            return;
        }
        if (atom.observations == 0) {
            atom.observations = 1;
        }
        atoms_.push_back(std::move(atom));
    }

    long double compression_ratio(const MentalState& state) const {
        const auto hits = activate(state, 1);
        if (hits.empty()) {
            return 1.0L;
        }
        return std::clamp(1.0L - 0.42L * hits.front().score, 0.08L, 1.0L);
    }

    std::vector<ConceptHit> activate(const MentalState& state, std::size_t limit) const {
        std::vector<ConceptHit> hits;
        for (const auto& atom : atoms_) {
            MentalState atom_state;
            atom_state.vectors = atom.centroid;
            atom_state.symbols = atom.symbols;
            const long double score = 0.55L * mental_similarity(state, atom_state) +
                                      0.35L * token_symbol_overlap(state, atom) +
                                      std::min<long double>(0.10L, atom.mdl_gain * 0.06L);
            if (score > 0.30L) {
                hits.push_back({atom, std::clamp(score, 0.0L, 1.0L)});
            }
        }
        std::sort(hits.begin(), hits.end(), [](const auto& left, const auto& right) {
            return left.score > right.score;
        });
        if (hits.size() > limit) {
            hits.resize(limit);
        }
        return hits;
    }

private:
    void observe_state(std::string name,
                       const MentalState& state,
                       long double reward,
                       long double free_energy) {
        if (state.vectors.empty()) {
            return;
        }
        const auto found = std::max_element(atoms_.begin(), atoms_.end(), [&](const auto& left, const auto& right) {
            MentalState left_state;
            left_state.vectors = left.centroid;
            left_state.symbols = left.symbols;
            MentalState right_state;
            right_state.vectors = right.centroid;
            right_state.symbols = right.symbols;
            return mental_similarity(state, left_state) < mental_similarity(state, right_state);
        });
        if (found != atoms_.end()) {
            MentalState found_state;
            found_state.vectors = found->centroid;
            found_state.symbols = found->symbols;
            if (mental_similarity(state, found_state) > 0.62L || atoms_.size() >= max_atoms_) {
                update_atom(*found, state, reward, free_energy);
                return;
            }
        }
        ConceptAtom atom;
        atom.name = concept_name(std::move(name), state);
        atom.centroid = state.vectors;
        atom.symbols = state.symbols;
        if (!state.goals.empty()) {
            for (auto snippet : clean_evidence_snippets(state.goals.front(), 4)) {
                atom.evidence_snippets.push_back(std::move(snippet));
            }
        }
        atom.observations = 1;
        atom.mdl_gain = std::max(0.05L, reward - free_energy);
        atoms_.push_back(std::move(atom));
    }

    static void update_atom(ConceptAtom& atom,
                            const MentalState& state,
                            long double reward,
                            long double free_energy) {
        ++atom.observations;
        const long double rate = 1.0L / static_cast<long double>(atom.observations);
        const std::size_t n = std::min(atom.centroid.size(), state.vectors.size());
        for (std::size_t i = 0; i < n; ++i) {
            atom.centroid[i] = (1.0L - rate) * atom.centroid[i] + rate * state.vectors[i];
        }
        normalize(atom.centroid);
        for (const auto& symbol : state.symbols) {
            if (std::find(atom.symbols.begin(), atom.symbols.end(), symbol) == atom.symbols.end()) {
                atom.symbols.push_back(symbol);
            }
        }
        if (!state.goals.empty() && atom.evidence_snippets.size() < 10U) {
            for (auto snippet : clean_evidence_snippets(state.goals.front(), 4)) {
                if (atom.evidence_snippets.size() >= 10U) {
                    break;
                }
                if (!snippet.empty() &&
                    std::find(atom.evidence_snippets.begin(), atom.evidence_snippets.end(), snippet) == atom.evidence_snippets.end()) {
                    atom.evidence_snippets.push_back(std::move(snippet));
                }
            }
        }
        atom.mdl_gain = std::max(atom.mdl_gain, reward - free_energy) +
                        0.02L * static_cast<long double>(atom.observations);
    }

    static std::string clean_evidence_snippet(std::string_view text) {
        const auto snippets = clean_evidence_snippets(text, 1);
        return snippets.empty() ? std::string{} : snippets.front();
    }

    static std::vector<std::string> clean_evidence_snippets(std::string_view text, std::size_t limit) {
        std::string normalized;
        normalized.reserve(std::min<std::size_t>(text.size(), 512U));
        bool last_space = false;
        for (const unsigned char ch : text) {
            if (std::isspace(ch) != 0) {
                if (!last_space) {
                    normalized.push_back(' ');
                    last_space = true;
                }
            } else {
                normalized.push_back(static_cast<char>(ch));
                last_space = false;
            }
            if (normalized.size() >= 512U) {
                break;
            }
        }
        const auto sentences = split_sentences(normalized);
        std::vector<std::string> out;
        for (auto sentence : sentences) {
            const auto lower = lower_copy(sentence);
            if (sentence.size() < 32U || lower.find("pmcid") != std::string::npos ||
                lower.find("author declared") != std::string::npos ||
                lower.find("conflict of interest") != std::string::npos ||
                !looks_like_sentence(sentence)) {
                continue;
            }
            if (sentence.size() > 220U) {
                sentence = truncate_sentence(std::move(sentence), 220U);
            }
            out.push_back(sentence);
            if (out.size() >= limit) {
                break;
            }
        }
        return out;
    }

    static std::vector<std::string> split_sentences(std::string_view text) {
        std::vector<std::string> out;
        std::string current;
        for (const char ch : text) {
            current.push_back(ch);
            if (ch == '.' || ch == '!' || ch == '?') {
                current = trim_copy(current);
                if (!current.empty()) {
                    out.push_back(current);
                }
                current.clear();
            }
        }
        current = trim_copy(current);
        if (!current.empty()) {
            out.push_back(current);
        }
        return out;
    }

    static std::string trim_copy(std::string text) {
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return {};
        }
        const auto last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1U);
    }

    static std::string lower_copy(std::string_view text) {
        std::string out(text);
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return out;
    }

    static bool looks_like_sentence(std::string_view sentence) {
        if (sentence.empty()) {
            return false;
        }
        const unsigned char first = static_cast<unsigned char>(sentence.front());
        if (std::islower(first) != 0 || std::isdigit(first) != 0 || sentence.front() == ')' || sentence.front() == ',' ||
            sentence.front() == ';' || sentence.front() == ':') {
            return false;
        }
        int parens = 0;
        for (const char ch : sentence) {
            if (ch == '(') {
                ++parens;
            } else if (ch == ')') {
                --parens;
            }
            if (parens < 0) {
                return false;
            }
        }
        return parens == 0;
    }

    static std::string truncate_sentence(std::string sentence, std::size_t limit) {
        if (sentence.size() <= limit) {
            return sentence;
        }
        auto cut = sentence.find_last_of(" ,;:)]}", limit);
        if (cut == std::string::npos || cut < limit / 2U) {
            cut = limit;
        }
        sentence.resize(cut);
        sentence = trim_copy(sentence);
        while (!sentence.empty() &&
               (sentence.back() == ',' || sentence.back() == ';' || sentence.back() == ':' ||
                sentence.back() == '(' || sentence.back() == '[')) {
            sentence.pop_back();
            sentence = trim_copy(sentence);
        }
        if (!sentence.empty() && sentence.back() != '.' && sentence.back() != '!' && sentence.back() != '?') {
            sentence.push_back('.');
        }
        return sentence;
    }

    static std::string concept_name(std::string fallback, const MentalState& state) {
        if (fallback != "trace_initial" && fallback != "trace_final") {
            return fallback;
        }
        std::string out;
        for (const auto& symbol : state.symbols) {
            if (symbol.rfind("token:", 0) != 0) {
                continue;
            }
            const auto token = symbol.substr(6U);
            if (!concept_token_allowed(token)) {
                continue;
            }
            if (!out.empty()) {
                out.push_back('_');
            }
            out += token;
            if (out.size() > 48U) {
                break;
            }
        }
        return out.empty() ? fallback : out;
    }

    static bool concept_token_allowed(std::string_view token) {
        if (token.size() < 4U) {
            return false;
        }
        if (token.rfind("pmc", 0) == 0 || token.rfind("pmid", 0) == 0 ||
            token.rfind("doi", 0) == 0 || token.rfind("http", 0) == 0) {
            return false;
        }
        if (token == "module" || token == "recursive" || token == "loop" ||
            token == "trace" || token == "initial" || token == "final" ||
            token == "corpus" || token == "chunk" || token == "bytes" ||
            token == "rdrand" || token == "impulse" || token == "through" ||
            token == "using" || token == "with" || token == "from" ||
            token == "that" || token == "this") {
            return false;
        }
        return std::all_of(token.begin(), token.end(), [](unsigned char ch) {
            return std::isalpha(ch) != 0;
        });
    }

    static void normalize(std::vector<long double>& vector) {
        const long double norm = std::sqrt(std::inner_product(vector.begin(), vector.end(), vector.begin(), 0.0L));
        if (norm <= 1.0e-12L) {
            return;
        }
        for (auto& value : vector) {
            value /= norm;
        }
    }

    static long double token_symbol_overlap(const MentalState& state, const ConceptAtom& atom) {
        std::size_t query_tokens = 0;
        std::size_t shared = 0;
        for (const auto& symbol : state.symbols) {
            if (symbol.rfind("token:", 0) != 0) {
                continue;
            }
            ++query_tokens;
            if (std::find(atom.symbols.begin(), atom.symbols.end(), symbol) != atom.symbols.end()) {
                ++shared;
            }
        }
        if (query_tokens == 0) {
            return 0.0L;
        }
        return static_cast<long double>(shared) / static_cast<long double>(query_tokens);
    }

    std::vector<ConceptAtom> atoms_;
    std::size_t max_atoms_;
};

} // namespace dzeta
