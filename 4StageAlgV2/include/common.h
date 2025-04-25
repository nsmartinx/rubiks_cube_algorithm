#pragma once
#include <cstdint>
#include <vector>
#include <string>

// shared integer aliases
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;

// the raw‐state globals (set in main, read by the packers in solvers.cu)
extern u64 g_rawEdgeState;
extern u64 g_rawCornerState;

// stage solver declarations
std::vector<std::string> solveStage1();
std::vector<std::string> solveStage2();
std::vector<std::string> solveStage3();
std::vector<std::string> solveStage4();
