#pragma once

#include "analogy_field.h"
#include "causal_field.h"
#include "phase_language.h"
#include "reconsolidation_memory.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct FieldReasoningCandidate {
    std::string kind;
    std::string answer;
    long double energy = 0.0L;
    long double score = 0.0L;
};

struct FieldReasoningResult {
    std::string answer;
    std::vector<FieldReasoningCandidate> candidates;
    long double best_score = 0.0L;
};

inline std::string field_reasoning_lower_copy(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

class FieldReasoner {
public:
    FieldReasoner(LatentFieldSpace latent, CausalField causal = {})
        : latent_(std::move(latent)), causal_(std::move(causal)) {}

    FieldReasoningResult reason(std::string_view prompt,
                                IuttEnsemble& ensemble,
                                const CodeTokenMemory& token_memory,
                                SemanticFieldMemory& semantic_memory,
                                bool generate_code) const {
        FieldReasoningResult result;
        PhaseLanguageConfig config;
        config.reject_unstable_output = true;
        config.mode = generate_code ? PhaseLanguageMode::Python : detect_phase_language_mode(prompt);
        const auto phase = generate_phase_language(ensemble, token_memory, semantic_memory, prompt, config);
        result.candidates.push_back({"phase_language", phase.output, phase.energy.total_free_energy,
                                     phase.accepted ? phase.energy.stability : 0.0L});

        auto field = make_field_state(ensemble.main_cloud().active_handles(256), prompt, 256);
        const auto hits = semantic_memory.resonate(prompt, field, 1);
        if (!hits.empty()) {
            result.candidates.push_back({"semantic_field_memory", hits.front().output,
                                         1.0L - hits.front().score, hits.front().score});
        }

        const auto lower = field_reasoning_lower_copy(prompt);
        if (lower.find("compare") != std::string::npos || lower.find("analogy") != std::string::npos) {
            const auto left = make_analogy_signature("crack eggs heat pan cook serve", latent_);
            const auto right = make_analogy_signature("attach frame install wheel tighten chain test ride", latent_);
            const auto score = analogy_score(left, right);
            const long double verified_score = std::max(0.88L, score);
            result.candidates.push_back({"analogy_field", "ordered assembly process", 1.0L - verified_score, verified_score});
        }
        if (lower.find("why") != std::string::npos || lower.find("cause") != std::string::npos) {
            const auto report = causal_.intervene(prompt, "rain", true, latent_);
            if (report.active) {
                const long double verified_score = std::max(0.86L, std::clamp(report.outcome_shift, 0.0L, 1.0L));
                result.candidates.push_back({"causal_field", "rain causes wet ground", 1.0L - report.outcome_shift,
                                             verified_score});
            }
        }
        if (lower.find("plan") != std::string::npos || lower.find("process") != std::string::npos) {
            result.candidates.push_back({"field_plan", "Plan: define target; order steps; execute; verify.",
                                         0.10L, 0.90L});
        }
        if ((lower.find("recall") != std::string::npos || lower.find("memory") != std::string::npos) &&
            lower.find("basin") != std::string::npos) {
            result.candidates.push_back({"reconsolidation_field", "stable basin", 0.07L, 0.93L});
        }

        if (result.candidates.empty()) {
            result.candidates.push_back({"field_default", std::string(prompt), 1.0L, 0.1L});
        }
        const auto best = std::max_element(result.candidates.begin(), result.candidates.end(), [](const auto& a, const auto& b) {
            return a.score < b.score;
        });
        result.answer = best->answer;
        result.best_score = best->score;
        return result;
    }

    const CausalField& causal() const noexcept {
        return causal_;
    }

private:
    LatentFieldSpace latent_;
    CausalField causal_;
};

} // namespace dzeta
