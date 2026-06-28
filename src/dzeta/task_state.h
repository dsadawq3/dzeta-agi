#pragma once

#include "iutt.h"

#include <string>
#include <vector>

namespace dzeta {

struct TaskState {
    std::string goal;
    std::string domain;
    std::vector<std::string> tokens;
    std::vector<std::string> constraints;
    std::vector<std::string> working_memory;
    std::vector<std::string> trace;
    IuttResult resonance;
};

} // namespace dzeta
