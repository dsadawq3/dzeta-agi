#pragma once

#include "latent_field_space.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct CausalFieldEdge {
    std::string cause;
    std::string effect;
    LatentVector cause_latent;
    LatentVector effect_latent;
    long double strength = 0.0L;
};

struct CausalInterventionReport {
    std::string cause;
    std::string effect;
    long double baseline = 0.0L;
    long double intervened = 0.0L;
    long double outcome_shift = 0.0L;
    bool active = false;
};

class CausalField {
public:
    void observe(std::string cause, std::string effect, const LatentFieldSpace& latent, long double strength = 0.5L) {
        for (auto& edge : edges_) {
            if (edge.cause == cause && edge.effect == effect) {
                edge.strength = std::clamp(edge.strength + 0.15L * strength, 0.0L, 1.0L);
                return;
            }
        }
        CausalFieldEdge edge;
        edge.cause = std::move(cause);
        edge.effect = std::move(effect);
        edge.cause_latent = latent.encode(edge.cause);
        edge.effect_latent = latent.encode(edge.effect);
        edge.strength = std::clamp(strength, 0.0L, 1.0L);
        edges_.push_back(std::move(edge));
    }

    CausalInterventionReport intervene(std::string_view effect,
                                       std::string_view cause,
                                       bool do_activate,
                                       const LatentFieldSpace& latent) const {
        const auto cause_query = latent.encode(cause);
        const auto effect_query = latent.encode(effect);
        CausalInterventionReport report;
        report.cause = std::string(cause);
        report.effect = std::string(effect);
        for (const auto& edge : edges_) {
            const long double cause_match = 0.5L + 0.5L * latent_cosine(edge.cause_latent, cause_query);
            const long double effect_match = 0.5L + 0.5L * latent_cosine(edge.effect_latent, effect_query);
            const long double contribution = edge.strength * cause_match * effect_match;
            report.baseline += 0.35L * contribution;
            report.intervened += (do_activate ? 1.0L : -1.0L) * contribution;
        }
        report.outcome_shift = std::abs(report.intervened - report.baseline);
        report.active = report.outcome_shift > 0.05L;
        return report;
    }

    void learn_intervention(std::string_view cause,
                            std::string_view effect,
                            const CausalInterventionReport& report,
                            bool confirmed) {
        for (auto& edge : edges_) {
            if (edge.cause == cause && edge.effect == effect) {
                const long double delta = confirmed ? 0.12L * report.outcome_shift : -0.08L;
                edge.strength = std::clamp(edge.strength + delta, 0.0L, 1.0L);
            }
        }
    }

    long double edge_strength(std::string_view cause, std::string_view effect) const {
        for (const auto& edge : edges_) {
            if (edge.cause == cause && edge.effect == effect) {
                return edge.strength;
            }
        }
        return 0.0L;
    }

    const std::vector<CausalFieldEdge>& edges() const noexcept {
        return edges_;
    }

private:
    std::vector<CausalFieldEdge> edges_;
};

} // namespace dzeta
