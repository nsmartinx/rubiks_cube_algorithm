#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>
#include "common.h"

uint64_t g_rawEdgeState;
uint64_t g_rawCornerState;

// Forward declarations for each solving stage
std::vector<std::string> solveStage1();
std::vector<std::string> solveStage2();
std::vector<std::string> solveStage3();
std::vector<std::string> solveStage4();

// A single quarter-turn move at the cubie level
struct RawMove {
    int cornerPermutation[8];
    int cornerOrientation[8];
    int edgePermutation[12];
    uint16_t edgeOrientation;
};

RawMove rawMoves[18] = {
    {{3,0,1,2,4,5,6,7},{0,0,0,0,0,0,0,0},{3,0,1,2,4,5,6,7,8,9,10,11},0b000000000000},
    {{2,3,0,1,4,5,6,7},{0,0,0,0,0,0,0,0},{2,3,0,1,4,5,6,7,8,9,10,11},0b000000000000},
    {{1,2,3,0,4,5,6,7},{0,0,0,0,0,0,0,0},{1,2,3,0,4,5,6,7,8,9,10,11},0b000000000000},

    {{4,1,2,0,7,5,6,3},{2,0,0,1,1,0,0,2},{8,1,2,3,11,5,6,7,4,9,10,0},0b000000000000},
    {{7,1,2,4,3,5,6,0},{0,0,0,0,0,0,0,0},{4,1,2,3,0,5,6,7,11,9,10,8},0b000000000000},
    {{3,1,2,7,0,5,6,4},{2,0,0,1,1,0,0,2},{11,1,2,3,8,5,6,7,0,9,10,4},0b000000000000},

    {{1,5,2,3,0,4,6,7},{1,2,0,0,2,1,0,0},{0,9,2,3,4,8,6,7,1,5,10,11},0b001100100010},
    {{5,4,2,3,1,0,6,7},{0,0,0,0,0,0,0,0},{0,5,2,3,4,1,6,7,9,8,10,11},0b000000000000},
    {{4,0,2,3,5,1,6,7},{1,2,0,0,2,1,0,0},{0,8,2,3,4,9,6,7,5,1,10,11},0b001100100010},

    {{0,1,2,3,5,6,7,4},{0,0,0,0,0,0,0,0},{0,1,2,3,5,6,7,4,8,9,10,11},0b000000000000},
    {{0,1,2,3,6,7,4,5},{0,0,0,0,0,0,0,0},{0,1,2,3,6,7,4,5,8,9,10,11},0b000000000000},
    {{0,1,2,3,7,4,5,6},{0,0,0,0,0,0,0,0},{0,1,2,3,7,4,5,6,8,9,10,11},0b000000000000},

    {{0,2,6,3,4,1,5,7},{0,1,2,0,0,2,1,0},{0,1,10,3,4,5,9,7,8,2,6,11},0b000000000000},
    {{0,6,5,3,4,2,1,7},{0,0,0,0,0,0,0,0},{0,1,6,3,4,5,2,7,8,10,9,11},0b000000000000},
    {{0,5,1,3,4,6,2,7},{0,1,2,0,0,2,1,0},{0,1,9,3,4,5,10,7,8,6,2,11},0b000000000000},

    {{0,1,3,7,4,5,2,6},{0,0,1,2,0,0,2,1},{0,1,2,11,4,5,6,10,8,9,3,7},0b110010001000},
    {{0,1,7,6,4,5,3,2},{0,0,0,0,0,0,0,0},{0,1,2,7,4,5,6,3,8,9,11,10},0b000000000000},
    {{0,1,6,2,4,5,7,3},{0,0,1,2,0,0,2,1},{0,1,2,10,4,5,6,11,8,9,7,3},0b110010001000}
};

const char* moveNotation[18] = {
    "U","U2","U'",
    "R","R2","R'",
    "F","F2","F'",
    "D","D2","D'",
    "L","L2","L'",
    "B","B2","B'"
};

int parseMoveToken(const std::string& token) {
    if (token.empty()) {
        return -1;
    }
    constexpr char const* faces = "URFDLB";
    auto position = std::string_view{faces}.find(token[0]);
    if (position == std::string_view::npos) {
        return -1;
    }
    int faceIndex = int(position);
    int turnOffset = 0;
    if (token.size() == 2) {
        turnOffset = (token[1] == '2' ? 1 : 2);
    }
    return faceIndex * 3 + turnOffset;
}

void applyRawMoveOnHost(uint64_t& cornerState,
                        uint64_t& edgeState,
                        int moveIndex) {
    const RawMove& move = rawMoves[moveIndex];
    uint64_t oldCorner = cornerState;
    uint64_t oldEdge = edgeState;
    uint64_t newCorner = 0;
    uint64_t newEdge = 0;

    for (int slot = 0; slot < 8; ++slot) {
        int source = move.cornerPermutation[slot];
        uint64_t packed = (oldCorner >> (5 * source)) & 0x1F;
        int pieceIndex = int(packed & 0x7);
        int orientation = int((packed >> 3) & 0x3);
        int twist = move.cornerOrientation[slot];
        int newOri = (orientation + twist) % 3;
        uint64_t outPacked = uint64_t(pieceIndex) | (uint64_t(newOri) << 3);
        newCorner |= outPacked << (5 * slot);
    }

    for (int slot = 0; slot < 12; ++slot) {
        int source = move.edgePermutation[slot];
        uint64_t packed = (oldEdge >> (5 * source)) & 0x1F;
        int pieceIndex = int(packed & 0xF);
        int orientation = int((packed >> 4) & 0x1);
        int flip = (move.edgeOrientation >> slot) & 1;
        int newOri = orientation ^ flip;
        uint64_t outPacked = uint64_t(pieceIndex) | (uint64_t(newOri) << 4);
        newEdge |= outPacked << (5 * slot);
    }

    cornerState = newCorner;
    edgeState = newEdge;
}

void flushAccumulatedMoves(std::vector<std::string>& output,
                           char& lastMoveFace,
                           int& turnCount) {
    if (!lastMoveFace) {
        return;
    }
    int remainder = turnCount % 4;
    if (remainder < 0) {
        remainder += 4;
    }
    if (remainder == 1) {
        output.push_back({lastMoveFace});
    } else if (remainder == 2) {
        output.push_back(std::string(1, lastMoveFace) + "2");
    } else if (remainder == 3) {
        output.push_back(std::string(1, lastMoveFace) + "'");
    }
    lastMoveFace = 0;
    turnCount = 0;
}

std::vector<std::string> splitString(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

int main() {
    std::vector<std::string> scramble = {
        "R2","D'","B'","R","B'","U2","B'","U'","L'",
        "U'","F'","L2","B2","F'","L2","B'","U","B",
        "F2","L2","B2","F'","L2","R'","U'"
    };

    uint64_t cornerState = 0;
    uint64_t edgeState = 0;
    for (int slotIndex = 0; slotIndex < 8; ++slotIndex) {
        cornerState |= uint64_t(slotIndex) << (5 * slotIndex);
    }
    for (int slotIndex = 0; slotIndex < 12; ++slotIndex) {
        edgeState |= uint64_t(slotIndex) << (5 * slotIndex);
    }

    std::cout << "Scrambling with:";
    for (auto& moveToken : scramble) {
        int moveIndex = parseMoveToken(moveToken);
        std::cout << " " << moveToken;
        applyRawMoveOnHost(cornerState, edgeState, moveIndex);
    }
    std::cout << "\n\n";

    g_rawCornerState = cornerState;
    g_rawEdgeState = edgeState;

    std::vector<std::string> completeSolution;

    auto stage1Solution = solveStage1();
    std::cout << "Stage 1:";
    for (auto& move : stage1Solution) {
        std::cout << " " << move;
        completeSolution.push_back(move);
        applyRawMoveOnHost(cornerState, edgeState, parseMoveToken(move));
    }
    std::cout << "\nLength: " << stage1Solution.size() << "\n\n";
    g_rawCornerState = cornerState;
    g_rawEdgeState = edgeState;

    auto stage2Solution = solveStage2();
    std::cout << "Stage 2:";
    for (auto& move : stage2Solution) {
        std::cout << " " << move;
        completeSolution.push_back(move);
        applyRawMoveOnHost(cornerState, edgeState, parseMoveToken(move));
    }
    std::cout << "\nLength: " << stage2Solution.size() << "\n\n";
    g_rawCornerState = cornerState;
    g_rawEdgeState = edgeState;

    auto stage3Solution = solveStage3();
    std::cout << "Stage 3:";
    for (auto& move : stage3Solution) {
        std::cout << " " << move;
        completeSolution.push_back(move);
        applyRawMoveOnHost(cornerState, edgeState, parseMoveToken(move));
    }
    std::cout << "\nLength: " << stage3Solution.size() << "\n\n";
    g_rawCornerState = cornerState;
    g_rawEdgeState = edgeState;

    auto stage4Solution = solveStage4();
    std::cout << "Stage 4:";
    for (auto& move : stage4Solution) {
        std::cout << " " << move;
        completeSolution.push_back(move);
        applyRawMoveOnHost(cornerState, edgeState, parseMoveToken(move));
    }
    std::cout << "\nLength: " << stage4Solution.size() << "\n\n";

    std::vector<std::string> finalSolution;
    char lastMoveFace = 0;
    int turnCount = 0;
    for (auto& move : completeSolution) {
        char faceChar = move[0];
        int turns = (move.size() == 2 ? (move[1] == '2' ? 2 : 3) : 1);
        if (faceChar != lastMoveFace) {
            flushAccumulatedMoves(finalSolution, lastMoveFace, turnCount);
            lastMoveFace = faceChar;
            turnCount = turns;
        } else {
            turnCount = (turnCount + turns) % 4;
        }
    }
    flushAccumulatedMoves(finalSolution, lastMoveFace, turnCount);

    std::cout << "Full solution:";
    for (auto& move : finalSolution) {
        std::cout << " " << move;
    }
    std::cout << "\nLength: " << finalSolution.size() << "\n";

    return 0;
}
