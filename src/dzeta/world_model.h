#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct Fact {
    std::string subject;
    std::string predicate;
    std::string object;
    std::string evidence;
    long double confidence = 1.0L;
};

struct CausalEdge {
    std::string cause;
    std::string effect;
    std::string evidence;
    long double weight = 1.0L;
};

struct FewShotExample {
    std::string input;
    std::string output;
    std::string domain;
};

struct Episode {
    std::string prompt;
    std::string answer;
    long double verification_score = 0.0L;
    bool success = false;
    std::string kind;
};

enum class LearnedOperatorKind {
    Unknown,
    Identity,
    Uppercase,
    Lowercase,
    Prefix,
    Suffix,
};

inline std::string world_lower_copy(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const unsigned char ch : text) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

inline std::string world_upper_copy(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const unsigned char ch : text) {
        out.push_back(static_cast<char>(std::toupper(ch)));
    }
    return out;
}

inline bool world_iequals(std::string_view left, std::string_view right) {
    return world_lower_copy(left) == world_lower_copy(right);
}

inline bool world_contains_ci(std::string_view haystack, std::string_view needle) {
    return world_lower_copy(haystack).find(world_lower_copy(needle)) != std::string::npos;
}

inline std::string learned_operator_kind_name(LearnedOperatorKind kind) {
    switch (kind) {
    case LearnedOperatorKind::Identity:
        return "identity";
    case LearnedOperatorKind::Uppercase:
        return "uppercase";
    case LearnedOperatorKind::Lowercase:
        return "lowercase";
    case LearnedOperatorKind::Prefix:
        return "prefix";
    case LearnedOperatorKind::Suffix:
        return "suffix";
    case LearnedOperatorKind::Unknown:
        return "unknown";
    }
    return "unknown";
}

inline LearnedOperatorKind parse_learned_operator_kind(std::string_view value) {
    if (value == "identity") {
        return LearnedOperatorKind::Identity;
    }
    if (value == "uppercase") {
        return LearnedOperatorKind::Uppercase;
    }
    if (value == "lowercase") {
        return LearnedOperatorKind::Lowercase;
    }
    if (value == "prefix") {
        return LearnedOperatorKind::Prefix;
    }
    if (value == "suffix") {
        return LearnedOperatorKind::Suffix;
    }
    return LearnedOperatorKind::Unknown;
}

struct LearnedOperator {
    LearnedOperatorKind kind = LearnedOperatorKind::Unknown;
    std::string domain = "general";
    std::string parameter;
    long double confidence = 0.0L;
    std::size_t observations = 0;

    std::optional<std::string> apply(std::string_view input) const {
        switch (kind) {
        case LearnedOperatorKind::Identity:
            return std::string(input);
        case LearnedOperatorKind::Uppercase:
            return world_upper_copy(input);
        case LearnedOperatorKind::Lowercase:
            return world_lower_copy(input);
        case LearnedOperatorKind::Prefix:
            return parameter + std::string(input);
        case LearnedOperatorKind::Suffix:
            return std::string(input) + parameter;
        case LearnedOperatorKind::Unknown:
            return std::nullopt;
        }
        return std::nullopt;
    }

    std::string name() const {
        auto out = learned_operator_kind_name(kind);
        if (!parameter.empty()) {
            out += "(" + parameter + ")";
        }
        return out;
    }
};

inline bool learned_domain_compatible(std::string_view learned, std::string_view requested) {
    return learned == requested || learned == "general" || requested == "general" ||
           learned == "text" || requested == "text";
}

inline std::optional<LearnedOperator> infer_learned_operator(std::string_view input,
                                                             std::string_view output,
                                                             std::string domain) {
    if (input.empty()) {
        return std::nullopt;
    }

    LearnedOperator op;
    op.domain = std::move(domain);
    op.confidence = 0.62L;
    op.observations = 1;

    if (output == input) {
        op.kind = LearnedOperatorKind::Identity;
        return op;
    }
    if (output == world_upper_copy(input)) {
        op.kind = LearnedOperatorKind::Uppercase;
        op.confidence = 0.74L;
        return op;
    }
    if (output == world_lower_copy(input)) {
        op.kind = LearnedOperatorKind::Lowercase;
        op.confidence = 0.74L;
        return op;
    }

    const std::string in(input);
    const std::string out(output);
    if (out.size() > in.size() && out.rfind(in, 0) == 0) {
        op.kind = LearnedOperatorKind::Suffix;
        op.parameter = out.substr(in.size());
        return op;
    }
    if (out.size() > in.size() && out.size() >= in.size() &&
        out.compare(out.size() - in.size(), in.size(), in) == 0) {
        op.kind = LearnedOperatorKind::Prefix;
        op.parameter = out.substr(0, out.size() - in.size());
        return op;
    }
    return std::nullopt;
}

class WorldModel {
public:
    void observe_fact(std::string subject,
                      std::string predicate,
                      std::string object,
                      std::string evidence,
                      long double confidence = 1.0L) {
        for (auto& fact : facts_) {
            const bool same_relation = world_iequals(fact.predicate, predicate) &&
                                       world_iequals(fact.object, object);
            const bool same_subject_relation = world_iequals(fact.subject, subject) &&
                                               world_iequals(fact.predicate, predicate);
            const bool contradiction = (same_relation && !world_iequals(fact.subject, subject)) ||
                                       (same_subject_relation && !world_iequals(fact.object, object));
            if (contradiction) {
                fact.confidence = std::max(0.0L, fact.confidence - 0.25L * confidence);
            }
        }
        facts_.push_back({std::move(subject), std::move(predicate), std::move(object),
                          std::move(evidence), std::clamp(confidence, 0.0L, 1.0L)});
    }

    void observe_cause(std::string cause, std::string effect, std::string evidence, long double weight = 1.0L) {
        for (auto& edge : causal_edges_) {
            if (world_iequals(edge.cause, cause) && world_iequals(edge.effect, effect)) {
                edge.weight = std::clamp(edge.weight + 0.15L * weight, 0.0L, 1.0L);
                if (!evidence.empty()) {
                    edge.evidence = std::move(evidence);
                }
                return;
            }
        }
        causal_edges_.push_back({std::move(cause), std::move(effect), std::move(evidence),
                                 std::clamp(weight, 0.0L, 1.0L)});
    }

    void observe_example(std::string input, std::string output, std::string domain = "general") {
        examples_.push_back({input, output, domain});
        auto inferred = infer_learned_operator(input, output, std::move(domain));
        if (inferred) {
            merge_operator(*inferred);
        }
    }

    void observe_episode(std::string prompt,
                         std::string answer,
                         long double verification_score,
                         bool success,
                         std::string kind) {
        episodes_.push_back({std::move(prompt), std::move(answer),
                             std::clamp(verification_score, 0.0L, 1.0L),
                             success, std::move(kind)});
        if (episodes_.size() > max_episodes_) {
            episodes_.erase(episodes_.begin(), episodes_.begin() +
                            static_cast<std::ptrdiff_t>(episodes_.size() - max_episodes_));
        }
    }

    std::optional<std::string> apply_learned_operator(std::string_view input,
                                                      std::string_view domain = "general") const {
        const LearnedOperator* best = nullptr;
        long double best_score = -1.0L;
        for (const auto& op : learned_operators_) {
            if (!learned_domain_compatible(op.domain, domain)) {
                continue;
            }
            const long double score = op.confidence + 0.03L * static_cast<long double>(op.observations);
            if (score > best_score) {
                best = &op;
                best_score = score;
            }
        }
        return best == nullptr ? std::nullopt : best->apply(input);
    }

    const std::vector<Fact>& facts() const noexcept {
        return facts_;
    }

    const std::vector<CausalEdge>& causes() const noexcept {
        return causal_edges_;
    }

    const std::vector<FewShotExample>& examples() const noexcept {
        return examples_;
    }

    const std::vector<Episode>& episodes() const noexcept {
        return episodes_;
    }

    const std::vector<LearnedOperator>& learned_operators() const noexcept {
        return learned_operators_;
    }

    void add_fact(Fact fact) {
        facts_.push_back(std::move(fact));
    }

    void add_cause(CausalEdge edge) {
        causal_edges_.push_back(std::move(edge));
    }

    void add_example(FewShotExample example) {
        examples_.push_back(std::move(example));
    }

    void add_episode(Episode episode) {
        episodes_.push_back(std::move(episode));
    }

    void add_learned_operator(LearnedOperator op) {
        merge_operator(std::move(op));
    }

private:
    void merge_operator(LearnedOperator op) {
        if (op.kind == LearnedOperatorKind::Unknown) {
            return;
        }
        for (auto& existing : learned_operators_) {
            if (existing.kind == op.kind && existing.parameter == op.parameter &&
                learned_domain_compatible(existing.domain, op.domain)) {
                existing.observations += std::max<std::size_t>(1, op.observations);
                existing.confidence = std::clamp((existing.confidence + op.confidence) * 0.5L + 0.08L,
                                                 0.0L, 1.0L);
                if (existing.domain == "general") {
                    existing.domain = op.domain;
                }
                return;
            }
        }
        learned_operators_.push_back(std::move(op));
    }

    std::vector<Fact> facts_;
    std::vector<CausalEdge> causal_edges_;
    std::vector<FewShotExample> examples_;
    std::vector<Episode> episodes_;
    std::vector<LearnedOperator> learned_operators_;
    std::size_t max_episodes_ = 4096;
};

inline char world_hex_digit(unsigned int value) {
    return static_cast<char>(value < 10 ? ('0' + value) : ('a' + (value - 10)));
}

inline std::string world_hex_encode(std::string_view text) {
    std::string encoded;
    encoded.reserve(text.size() * 2);
    for (unsigned char ch : text) {
        encoded.push_back(world_hex_digit((ch >> 4U) & 0x0FU));
        encoded.push_back(world_hex_digit(ch & 0x0FU));
    }
    return encoded;
}

inline unsigned int world_hex_value(char ch) {
    if (ch >= '0' && ch <= '9') {
        return static_cast<unsigned int>(ch - '0');
    }
    if (ch >= 'a' && ch <= 'f') {
        return static_cast<unsigned int>(ch - 'a' + 10);
    }
    if (ch >= 'A' && ch <= 'F') {
        return static_cast<unsigned int>(ch - 'A' + 10);
    }
    throw std::runtime_error("invalid world model hex byte");
}

inline std::string world_hex_decode(std::string_view encoded) {
    if (encoded.size() % 2 != 0) {
        throw std::runtime_error("odd-length world model hex field");
    }
    std::string decoded;
    decoded.reserve(encoded.size() / 2);
    for (std::size_t i = 0; i < encoded.size(); i += 2) {
        const auto high = world_hex_value(encoded[i]);
        const auto low = world_hex_value(encoded[i + 1]);
        decoded.push_back(static_cast<char>((high << 4U) | low));
    }
    return decoded;
}

inline std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    for (;;) {
        const auto tab = line.find('\t', begin);
        if (tab == std::string::npos) {
            fields.push_back(line.substr(begin));
            return fields;
        }
        fields.push_back(line.substr(begin, tab - begin));
        begin = tab + 1;
    }
}

inline void save_world_model(const WorldModel& model, std::string_view path) {
    std::ofstream output(std::string(path), std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write world model: " + std::string(path));
    }
    output << "DZETA_WORLD_MODEL_V1\n";
    for (const auto& fact : model.facts()) {
        output << "fact\t" << world_hex_encode(fact.subject)
               << '\t' << world_hex_encode(fact.predicate)
               << '\t' << world_hex_encode(fact.object)
               << '\t' << world_hex_encode(fact.evidence)
               << '\t' << static_cast<double>(fact.confidence) << '\n';
    }
    for (const auto& edge : model.causes()) {
        output << "cause\t" << world_hex_encode(edge.cause)
               << '\t' << world_hex_encode(edge.effect)
               << '\t' << world_hex_encode(edge.evidence)
               << '\t' << static_cast<double>(edge.weight) << '\n';
    }
    for (const auto& example : model.examples()) {
        output << "example\t" << world_hex_encode(example.input)
               << '\t' << world_hex_encode(example.output)
               << '\t' << world_hex_encode(example.domain) << '\n';
    }
    for (const auto& episode : model.episodes()) {
        output << "episode\t" << world_hex_encode(episode.prompt)
               << '\t' << world_hex_encode(episode.answer)
               << '\t' << static_cast<double>(episode.verification_score)
               << '\t' << (episode.success ? 1 : 0)
               << '\t' << world_hex_encode(episode.kind) << '\n';
    }
    for (const auto& op : model.learned_operators()) {
        output << "operator\t" << learned_operator_kind_name(op.kind)
               << '\t' << world_hex_encode(op.domain)
               << '\t' << world_hex_encode(op.parameter)
               << '\t' << static_cast<double>(op.confidence)
               << '\t' << op.observations << '\n';
    }
}

inline WorldModel load_world_model(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read world model: " + std::string(path));
    }
    std::string header;
    std::getline(input, header);
    if (header != "DZETA_WORLD_MODEL_V1") {
        throw std::runtime_error("invalid world model header: " + std::string(path));
    }

    WorldModel model;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split_tabs(line);
        if (fields.empty()) {
            continue;
        }
        if (fields[0] == "fact" && fields.size() == 6) {
            model.add_fact({world_hex_decode(fields[1]), world_hex_decode(fields[2]),
                            world_hex_decode(fields[3]), world_hex_decode(fields[4]),
                            std::stold(fields[5])});
        } else if (fields[0] == "cause" && fields.size() == 5) {
            model.add_cause({world_hex_decode(fields[1]), world_hex_decode(fields[2]),
                             world_hex_decode(fields[3]), std::stold(fields[4])});
        } else if (fields[0] == "example" && fields.size() == 4) {
            model.add_example({world_hex_decode(fields[1]), world_hex_decode(fields[2]),
                               world_hex_decode(fields[3])});
        } else if (fields[0] == "episode" && fields.size() == 6) {
            model.add_episode({world_hex_decode(fields[1]), world_hex_decode(fields[2]),
                               std::stold(fields[3]), fields[4] == "1", world_hex_decode(fields[5])});
        } else if (fields[0] == "operator" && fields.size() == 6) {
            model.add_learned_operator({parse_learned_operator_kind(fields[1]),
                                        world_hex_decode(fields[2]),
                                        world_hex_decode(fields[3]),
                                        std::stold(fields[4]),
                                        static_cast<std::size_t>(std::stoull(fields[5]))});
        } else {
            throw std::runtime_error("invalid world model row: " + std::string(path));
        }
    }
    return model;
}

} // namespace dzeta
