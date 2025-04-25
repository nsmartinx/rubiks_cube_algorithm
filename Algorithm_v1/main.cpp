#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>

using uint64 = uint64_t;

// Forward declarations of our four-stage solvers
std::vector<std::string> solveStage1(uint64 edgeState);
std::vector<std::string> solveStage2(uint64 cornerState, uint64 edgeState);
std::vector<std::string> solveStage3(uint64 cornerState, uint64 edgeState);
std::vector<std::string> solveStage4(uint64 cornerState, uint64 edgeState);

// Raw quarter-turn moves at the cubie level
// Corner slots: 0=URF, 1=UFL, 2=ULB, 3=UBR, 4=DFR, 5=DLF, 6=DBL, 7=DRB
// Edge slots:   0=UR, 1=UF, 2=UL, 3=UB, 4=DR, 5=DF, 6=DL, 7=DB,
//               8=FR, 9=FL, 10=BL, 11=BR
struct RawMove {
    int cornerPermutation[8];     // Piece that ends up in slot i
    int cornerOrientation[8];     // Increase in rotation of the piece that ends in slot i
    int edgePermutation[12];      // Piece that ends up in slot i
    uint16_t edgeOrientation; // If Piece that ends up in slot i gets flipped
};

RawMove rawMoves[18] = {
    { // U
        {3, 0, 1, 2, 4, 5, 6, 7},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11},
        0b0000'0000'0000
    },
    { // U2
        {2, 3, 0, 1, 4, 5, 6, 7},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {2, 3, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11},
        0b0000'0000'0000
    },
    { // U'
        {1, 2, 3, 0, 4, 5, 6, 7},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11},
        0b0000'0000'0000
    },

    { // R
        {4, 1, 2, 0, 7, 5, 6, 3},
        {2, 0, 0, 1, 1, 0, 0, 2},
        {8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0},
        0b0000'0000'0000
    },
    { // R2
        {7, 1, 2, 4, 3, 5, 6, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {4, 1, 2, 3, 0, 5, 6, 7, 11, 9, 10, 8},
        0b0000'0000'0000
    },
    { // R'
        {3, 1, 2, 7, 0, 5, 6, 4},
        {2, 0, 0, 1, 1, 0, 0, 2},
        {11, 1, 2, 3, 8, 5, 6, 7, 0, 9, 10, 4},
        0b0000'0000'0000
    },

    { // F
        {1, 5, 2, 3, 0, 4, 6, 7},
        {1, 2, 0, 0, 2, 1, 0, 0},
        {0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11},
        0b0011'0010'0010
    },
    { // F2
        {5, 4, 2, 3, 1, 0, 6, 7},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 5, 2, 3, 4, 1, 6, 7, 9, 8, 10, 11},
        0b0000'0000'0000
    },
    { // F'
        {4, 0, 2, 3, 5, 1, 6, 7},
        {1, 2, 0, 0, 2, 1, 0, 0},
        {0, 8, 2, 3, 4, 9, 6, 7, 5, 1, 10, 11},
        0b0011'0010'0010
    },

    { // D
        {0, 1, 2, 3, 5, 6, 7, 4},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11},
        0b0000'0000'0000
    },
    { // D2
        {0, 1, 2, 3, 6, 7, 4, 5},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11},
        0b0000'0000'0000
    },
    { // D'
        {0, 1, 2, 3, 7, 4, 5, 6},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11},
        0b0000'0000'0000
    },

    { // L
        {0, 2, 6, 3, 4, 1, 5, 7},
        {0, 1, 2, 0, 0, 2, 1, 0},
        {0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11},
        0b0000'0000'0000
    },
    { // L2
        {0, 6, 5, 3, 4, 2, 1, 7},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 6, 3, 4, 5, 2, 7, 8, 10, 9, 11},
        0b0000'0000'0000
    },
    { // L'
        {0, 5, 1, 3, 4, 6, 2, 7},
        {0, 1, 2, 0, 0, 2, 1, 0},
        {0, 1, 9, 3, 4, 5, 10, 7, 8, 6, 2, 11},
        0b0000'0000'0000
    },

    { // B
        {0, 1, 3, 7, 4, 5, 2, 6},
        {0, 0, 1, 2, 0, 0, 2, 1},
        {0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7},
        0b1100'1000'1000
    },
    { // B2
        {0, 1, 7, 6, 4, 5, 3, 2},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 7, 4, 5, 6, 3, 8, 9, 11, 10},
        0b0000'0000'0000
    },
    { // B'
        {0, 1, 6, 2, 4, 5, 7, 3},
        {0, 0, 1, 2, 0, 0, 2, 1},
        {0, 1, 2, 10, 4, 5, 6, 11, 8, 9, 7, 3},
        0b1100'1000'1000
    }
};

const char* moveNotation[18] = {
    "U", "U2", "U'",
    "R", "R2", "R'",
    "F", "F2", "F'",
    "D", "D2", "D'",
    "L", "L2", "L'",
    "B", "B2", "B'"
};

// Convert a notation token to a move index
int parseMoveToken(const std::string& token) {
    if (token.empty()) {
        return -1;
    }
    int faceIndex = 0;
    switch (token[0]) {
        case 'U': faceIndex = 0; break;
        case 'R': faceIndex = 1; break;
        case 'F': faceIndex = 2; break;
        case 'D': faceIndex = 3; break;
        case 'L': faceIndex = 4; break;
        case 'B': faceIndex = 5; break;
        default:    return -1;
    }
    int turnOffset = 0;
    if (token.size() == 2) {
        turnOffset = (token[1] == '2' ? 1 : 2);
    }
    return faceIndex * 3 + turnOffset;
}

// Apply a raw move on the CPU
void applyRawMoveOnCPU(uint64& cornerState, uint64& edgeState, int moveIndex) {
    const RawMove& move = rawMoves[moveIndex];
    uint64 oldCorners = cornerState;
    uint64 oldEdges = edgeState;
    uint64 newCorners = 0;
    uint64 newEdges = 0;

    // Apply corner permutation and orientation
    for (int dest = 0; dest < 8; ++dest) {
        int src = move.cornerPermutation[dest];
        uint64 packed = (oldCorners >> (5 * src)) & 0x1F;
        int pieceIndex = packed & 0x7;
        int orientation = (packed >> 3) & 0x3;
        int newOrientation = (orientation + move.cornerOrientation[dest]) % 3;
        uint64 outPacked = uint64(pieceIndex) | (uint64(newOrientation) << 3);
        newCorners |= outPacked << (5 * dest);
    }

    // Apply edge permutation and orientation
    for (int dest = 0; dest < 12; ++dest) {
        int src = move.edgePermutation[dest];
        uint64 packed = (oldEdges >> (5 * src)) & 0x1F;
        int pieceIndex = packed & 0xF;
        int orientation = (packed >> 4) & 0x1;
        int newOrientation = orientation ^ ((move.edgeOrientation >> dest) & 1);
        uint64 outPacked = uint64(pieceIndex) | (uint64(newOrientation) << 4);
        newEdges |= outPacked << (5 * dest);
    }

    cornerState = newCorners;
    edgeState = newEdges;
}

// Flush accumulated turns for one face into simplifiedMoves
void flushAccumulatedMoves(std::vector<std::string>& simplifiedMoves,
                           char& lastFace,
                           int& turnsAccumulated) {
    if (lastFace == 0) {
        return;
    }
    int r = turnsAccumulated % 4;
    if (r < 0) {
        r += 4;
    }
    if (r == 1) {
        simplifiedMoves.push_back(std::string(1, lastFace));
    } else if (r == 2) {
        simplifiedMoves.push_back(std::string(1, lastFace) + "2");
    } else if (r == 3) {
        simplifiedMoves.push_back(std::string(1, lastFace) + "'");
    }
    lastFace = 0;
    turnsAccumulated = 0;
}

// Split a string by whitespace into tokens
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
    // Hardcoded scramble sequence
    std::vector<std::string> scrambleMoves = {
        "R2", "D'", "B'", "R", "B'", "U2", "B'", "U'", "L'",
        "U'", "F'", "L2", "B2", "F'", "L2", "B'", "U", "B",
        "F2", "L2", "B2", "F'", "L2", "R'", "U'"
    };

    // Initialize cube to solved state
    uint64 cornerState = 0;
    uint64 edgeState = 0;
    for (int i = 0; i < 8; ++i) {
        cornerState |= (uint64(i) << (5 * i));
    }
    for (int i = 0; i < 12; ++i) {
        edgeState |= (uint64(i) << (5 * i));
    }

    // Apply scramble
    std::cout << "Scrambling with:";
    for (const auto& mv : scrambleMoves) {
        int idx = parseMoveToken(mv);
        if (idx < 0) {
            std::cerr << "\nError: invalid move '" << mv << "'\n";
            return 1;
        }
        std::cout << " " << mv;
        applyRawMoveOnCPU(cornerState, edgeState, idx);
    }
    std::cout << "\n\n";

    // Four-stage solve
    std::vector<std::string> totalSolution;

    // Stage 1
    auto sol1 = solveStage1(edgeState);
    std::cout << "Stage 1 solution:";
    for (const auto& mv : sol1) {
        std::cout << " " << mv;
        totalSolution.push_back(mv);
        applyRawMoveOnCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout << "\nLength: " << sol1.size() << "\n\n";

    // Stage 2
    auto sol2 = solveStage2(cornerState, edgeState);
    std::cout << "Stage 2 solution:";
    for (const auto& mv : sol2) {
        std::cout << " " << mv;
        totalSolution.push_back(mv);
        applyRawMoveOnCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout << "\nLength: " << sol2.size() << "\n\n";

    // Stage 3
    auto sol3 = solveStage3(cornerState, edgeState);
    std::cout << "Stage 3 solution:";
    for (const auto& mv : sol3) {
        std::cout << " " << mv;
        totalSolution.push_back(mv);
        applyRawMoveOnCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout << "\nLength: " << sol3.size() << "\n\n";

    // Stage 4
    auto sol4 = solveStage4(cornerState, edgeState);
    std::cout << "Stage 4 solution:";
    for (const auto& mv : sol4) {
        std::cout << " " << mv;
        totalSolution.push_back(mv);
        applyRawMoveOnCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout << "\nLength: " << sol4.size() << "\n\n";

    // Simplify solution
    std::vector<std::string> simplified;
    char lastFace = 0;
    int accumulated = 0;

    for (const auto& mv : totalSolution) {
        char face = mv[0];
        int turns = (mv.size() == 2 ? (mv[1] == '2' ? 2 : 3) : 1);
        if (face != lastFace) {
            flushAccumulatedMoves(simplified, lastFace, accumulated);
            lastFace = face;
            accumulated = turns;
        } else {
            accumulated = (accumulated + turns) % 4;
        }
    }
    flushAccumulatedMoves(simplified, lastFace, accumulated);

    // Output full simplified solution
    std::cout << "Full solution:";
    for (const auto& mv : simplified) {
        std::cout << " " << mv;
    }
    std::cout << "\nLength: " << simplified.size() << "\n";

    return 0;
}
