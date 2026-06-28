#pragma once

#include "mentalese_core.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dzeta {

struct EvidenceHit {
    std::string text;
    long double score = 0.0L;
};

struct EvidenceSentence {
    std::string text;
    MentalState state;
    std::vector<std::string> tokens;
    std::size_t observations = 1;
    long double reliability = 0.5L;
};

class SemanticEvidenceMemory {
public:
    explicit SemanticEvidenceMemory(std::size_t max_sentences = 4096,
                                    std::size_t dimension = 32)
        : max_sentences_(std::max<std::size_t>(32, max_sentences)),
          dimension_(std::max<std::size_t>(1, dimension)) {}

    void observe_text(std::string_view text) {
        for (auto sentence : prepare_sentences(text, dimension_, 12)) {
            observe_prepared_sentence(std::move(sentence));
        }
        compact();
    }

    void observe_sentence(std::string sentence) {
        if (!clean_generated_sentence(sentence)) {
            return;
        }
        auto item = prepare_sentence(std::move(sentence), dimension_);
        if (item.text.empty()) {
            return;
        }
        observe_prepared_sentence(std::move(item));
        compact();
    }

    void observe_prepared_sentence(EvidenceSentence item) {
        if (item.text.empty()) {
            return;
        }
        sentences_.push_back(std::move(item));
        if (sentences_.size() > max_sentences_ * 2U) {
            compact();
        }
    }

    void restore_serialized_sentence(std::string text,
                                     std::size_t observations,
                                     long double reliability) {
        auto item = prepare_sentence(std::move(text), dimension_);
        if (item.text.empty()) {
            return;
        }
        item.observations = std::max<std::size_t>(1, observations);
        item.reliability = std::clamp(reliability, 0.0L, 1.0L);
        observe_prepared_sentence(std::move(item));
    }

    void compact() {
        if (sentences_.size() <= max_sentences_) {
            return;
        }
        prune();
    }

    std::vector<EvidenceHit> retrieve(const MentalState& query,
                                      const std::vector<std::string>& query_tokens,
                                      std::size_t limit) const {
        std::vector<EvidenceHit> hits;
        for (const auto& sentence : sentences_) {
            const long double score = score_sentence(sentence, query, query_tokens);
            if (score >= 0.34L) {
                hits.push_back({sentence.text, std::clamp(score, 0.0L, 1.0L)});
            }
        }
        std::sort(hits.begin(), hits.end(), [](const auto& left, const auto& right) {
            return left.score > right.score;
        });
        std::vector<EvidenceHit> out;
        for (const auto& hit : hits) {
            if (std::none_of(out.begin(), out.end(), [&](const auto& kept) {
                    return sentence_redundancy(kept.text, hit.text) > 0.72L;
                })) {
                out.push_back(hit);
                if (out.size() >= limit) {
                    break;
                }
            }
        }
        return out;
    }

    std::size_t size() const noexcept {
        return sentences_.size();
    }

    const std::vector<EvidenceSentence>& sentences() const noexcept {
        return sentences_;
    }

    void reserve(std::size_t count) {
        sentences_.reserve(count);
    }

    static std::vector<std::string> extract_clean_sentences(std::string_view text, std::size_t limit) {
        std::string normalized;
        normalized.reserve(std::min<std::size_t>(text.size(), 2048U));
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
        }
        std::vector<std::string> out;
        std::string current;
        for (const char ch : normalized) {
            current.push_back(ch);
            if (ch == '.' || ch == '!' || ch == '?') {
                current = trim(std::move(current));
                if (clean_generated_sentence(current)) {
                    out.push_back(truncate_sentence(std::move(current), 260U));
                    if (out.size() >= limit) {
                        break;
                    }
                }
                current.clear();
            }
        }
        current = trim(std::move(current));
        if (out.size() < limit && clean_generated_sentence(current)) {
            out.push_back(truncate_sentence(std::move(current), 260U));
        }
        return out;
    }

    static EvidenceSentence prepare_sentence(std::string sentence, std::size_t dimension) {
        EvidenceSentence item;
        if (!clean_generated_sentence(sentence)) {
            return item;
        }
        item.text = std::move(sentence);
        item.state = encode_prompt_mental_state(item.text, dimension);
        item.tokens = tokenize_query(item.text);
        item.reliability = 0.58L;
        return item;
    }

    static std::vector<EvidenceSentence> prepare_sentences(std::string_view text,
                                                           std::size_t dimension,
                                                           std::size_t limit) {
        std::vector<EvidenceSentence> out;
        for (auto sentence : extract_clean_sentences(text, limit)) {
            auto prepared = prepare_sentence(std::move(sentence), dimension);
            if (!prepared.text.empty()) {
                out.push_back(std::move(prepared));
            }
        }
        return out;
    }

private:
    void prune() {
        std::sort(sentences_.begin(), sentences_.end(), [](const auto& left, const auto& right) {
            if (left.reliability == right.reliability) {
                return left.observations > right.observations;
            }
            return left.reliability > right.reliability;
        });
        sentences_.resize(max_sentences_);
    }

    static long double score_sentence(const EvidenceSentence& sentence,
                                      const MentalState& query,
                                      const std::vector<std::string>& query_tokens) {
        const long double semantic = mental_similarity(sentence.state, query);
        const long double overlap = token_overlap(sentence.tokens, query_tokens);
        const long double exact = exact_query_presence(sentence.text, query_tokens);
        return 0.45L * semantic + 0.35L * overlap + 0.12L * exact + 0.08L * sentence.reliability;
    }

    static long double token_overlap(const std::vector<std::string>& left,
                                     const std::vector<std::string>& right) {
        if (left.empty() || right.empty()) {
            return 0.0L;
        }
        std::size_t shared = 0;
        std::size_t considered = 0;
        for (const auto& token : right) {
            if (token.size() < 4U || !content_token(token)) {
                continue;
            }
            ++considered;
            if (std::find(left.begin(), left.end(), token) != left.end()) {
                ++shared;
            }
        }
        if (considered == 0) {
            return 0.0L;
        }
        return static_cast<long double>(shared) / static_cast<long double>(considered);
    }

    static long double exact_query_presence(const std::string& sentence,
                                            const std::vector<std::string>& query_tokens) {
        const auto lower = lower_copy(sentence);
        std::size_t hits = 0;
        std::size_t considered = 0;
        for (const auto& token : query_tokens) {
            if (token.size() < 4U || !content_token(token)) {
                continue;
            }
            ++considered;
            if (lower.find(token) != std::string::npos) {
                ++hits;
            }
        }
        if (considered == 0) {
            return 0.0L;
        }
        return static_cast<long double>(hits) / static_cast<long double>(considered);
    }

    static long double sentence_redundancy(std::string_view left, std::string_view right) {
        const auto left_tokens = tokenize_query(left);
        const auto right_tokens = tokenize_query(right);
        if (left_tokens.empty() || right_tokens.empty()) {
            return 0.0L;
        }
        std::size_t shared = 0;
        for (const auto& token : left_tokens) {
            if (std::find(right_tokens.begin(), right_tokens.end(), token) != right_tokens.end()) {
                ++shared;
            }
        }
        const auto denom = std::max(left_tokens.size(), right_tokens.size());
        return static_cast<long double>(shared) / static_cast<long double>(denom);
    }

    static bool clean_generated_sentence(const std::string& sentence) {
        const auto lower = lower_copy(sentence);
        if (sentence.size() < 32U || lower.find("pmcid") != std::string::npos ||
            lower.find(" pmc") != std::string::npos ||
            lower.find("author metadata") != std::string::npos ||
            lower.find("conflict of interest") != std::string::npos ||
            lower.find("module recursive") != std::string::npos ||
            lower.find("module:") != std::string::npos) {
            return false;
        }
        const unsigned char first = static_cast<unsigned char>(sentence.front());
        if (std::islower(first) != 0 || std::isdigit(first) != 0 ||
            sentence.front() == ')' || sentence.front() == ',' ||
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
        if (parens != 0) {
            return false;
        }
        return std::any_of(sentence.begin(), sentence.end(), [](unsigned char ch) {
            return std::isalpha(ch) != 0;
        });
    }

    static bool content_token(std::string_view token) {
        if (token.rfind("pmc", 0) == 0 || token == "explain" ||
            token == "describe" || token == "process" ||
            token == "what" || token == "this" || token == "that" ||
            token == "with" || token == "from") {
            return false;
        }
        return std::all_of(token.begin(), token.end(), [](unsigned char ch) {
            return std::isalnum(ch) != 0;
        });
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
        sentence = trim(std::move(sentence));
        if (!sentence.empty() && sentence.back() != '.' && sentence.back() != '!' && sentence.back() != '?') {
            sentence.push_back('.');
        }
        return sentence;
    }

    static std::string trim(std::string text) {
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

    std::vector<EvidenceSentence> sentences_;
    std::size_t max_sentences_;
    std::size_t dimension_;
};

} // namespace dzeta
