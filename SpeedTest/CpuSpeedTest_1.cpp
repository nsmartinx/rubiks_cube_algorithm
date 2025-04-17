#include <iostream>
#include <chrono>
#include <cstdint>
#include <vector>
#include <string>

using u64 = uint64_t;

// —— Configuration ——
// maximum brute‑force depth;
#define MAX_SEARCH_DEPTH 6

// —— Cubie‑level raw quarter‑turn moves ——
struct RawMove {
    int cornerPerm[8];
    int cornerOrientation[8];
    int edgePerm[12];
    int edgeOrientation[12];
};

RawMove quarterTurns[6] = {
    // U
    { {3,0,1,2,4,5,6,7}, {0,0,0,0,0,0,0,0},
      {3,0,1,2,4,5,6,7,8,9,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} },
    // R
    { {4,1,2,0,7,5,6,3}, {2,0,0,1,1,0,0,2},
      {8,1,2,3,11,5,6,7,4,9,10,0},    {0,0,0,0,0,0,0,0,0,0,0,0} },
    // F
    { {1,5,2,3,0,4,6,7}, {1,2,0,0,2,1,0,0},
      {0,9,2,3,4,8,6,7,1,5,10,11},    {0,1,0,0,0,1,0,0,1,1,0,0} },
    // D
    { {0,1,2,3,5,6,7,4}, {0,0,0,0,0,0,0,0},
      {0,1,2,3,5,6,7,4,8,9,10,11},    {0,0,0,0,0,0,0,0,0,0,0,0} },
    // L
    { {0,2,6,3,4,1,5,7}, {0,1,2,0,0,2,1,0},
      {0,1,10,3,4,5,9,7,8,2,6,11},    {0,0,0,0,0,0,0,0,0,0,0,0} },
    // B
    { {0,1,3,7,4,5,2,6}, {0,0,1,2,0,0,2,1},
      {0,1,2,11,4,5,6,10,8,9,3,7},    {0,0,0,1,0,0,0,1,0,0,1,1} }
};

// human‑readable move names
const char* moveNotationNames[18] = {
    "U","U2","U'","R","R2","R'",
    "F","F2","F'","D","D2","D'",
    "L","L2","L'","B","B2","B'"
};

// parse tokens to moves
int parseMoveToken(const std::string& token) {
    if (token.empty()) return -1;
    int faceIndex;
    switch (token[0]) {
        case 'U': faceIndex = 0; break;
        case 'R': faceIndex = 1; break;
        case 'F': faceIndex = 2; break;
        case 'D': faceIndex = 3; break;
        case 'L': faceIndex = 4; break;
        case 'B': faceIndex = 5; break;
        default:  return -1;
    }
    int turnType = 0;
    if (token.size() == 2) {
        if (token[1] == '2')       turnType = 1;
        else if (token[1] == '\'') turnType = 2;
        else return -1;
    }
    return faceIndex * 3 + turnType;
}

// inverse of a move
inline int inverseMoveIndex(int moveIndex) {
    int faceIndex = moveIndex / 3;
    int turnType  = moveIndex % 3;
    int inverseType = (turnType == 0 ? 2 : (turnType == 1 ? 1 : 0));
    return faceIndex * 3 + inverseType;
}

// global cube state
u64 cornersState, edgesState;
// solved state for comparison
u64 solvedCornersState, solvedEdgesState;
// flag to skip further checks after first solution
bool foundSolution = false;

// apply one quarter‑turn
void applyQuarterTurn(int face) {
    const RawMove& mv = quarterTurns[face];
    u64 oldCorners   = cornersState;
    u64 oldEdges     = edgesState;
    u64 newCorners   = 0;
    u64 newEdges     = 0;

    for (int i = 0; i < 8; ++i) {
        u64 packedData       = (oldCorners >> (5 * i)) & 0x1FULL;
        int pieceIndex       = packedData & 7;
        int orientation      = (packedData >> 3) & 3;
        int newOrientation   = (orientation + mv.cornerOrientation[i]) % 3;
        int destIndex        = mv.cornerPerm[i];
        u64 outPacked        = u64(pieceIndex) | (u64(newOrientation) << 3);
        newCorners          |= outPacked << (5 * destIndex);
    }

    for (int j = 0; j < 12; ++j) {
        u64 packedData       = (oldEdges >> (5 * j)) & 0x1FULL;
        int pieceIndex       = packedData & 0xF;
        int orientation      = (packedData >> 4) & 1;
        int newOrientation   = orientation ^ mv.edgeOrientation[j];
        int destIndex        = mv.edgePerm[j];
        u64 outPacked        = u64(pieceIndex) | (u64(newOrientation) << 4);
        newEdges            |= outPacked << (5 * destIndex);
    }

    cornersState = newCorners;
    edgesState   = newEdges;
}

// apply any of the 18 moves (turn=0: CW, 1: 180, 2: CCW)
void applyTurn(int move) {
    int faceIndex = move / 3;
    int turnType  = move % 3;
    int repeats   = (turnType == 1 ? 2 : (turnType == 2 ? 3 : 1));
    while (repeats--) applyQuarterTurn(faceIndex);
}

// initialize to solved state
void initCube() {
    cornersState = edgesState = 0;
    for (int i = 0; i < 8; ++i) cornersState |= (u64(i) << (5 * i));
    for (int j = 0; j < 12; ++j) edgesState   |= (u64(j) << (5 * j));
    solvedCornersState = cornersState;
    solvedEdgesState   = edgesState;
}

// record current path
int solutionMoves[MAX_SEARCH_DEPTH];

// print the found solution
void printSolution(int depth) {
    std::cout << "Solution found (" << depth << " moves): ";
    for (int i = 0; i < depth; ++i) {
        std::cout << moveNotationNames[solutionMoves[i]]
                  << (i + 1 < depth ? " " : "");
    }
    std::cout << "\n";
}

// depth‑limited brute force, prints first solution but continues exploring
void searchDepthLimited(int maxDepth, int depth = 0) {
    if (!foundSolution && depth > 0
        && cornersState == solvedCornersState
        && edgesState   == solvedEdgesState) {
        printSolution(depth);
        foundSolution = true;
    }
    if (depth == maxDepth) return;

    for (int move = 0; move < 18; ++move) {
        applyTurn(move);
        solutionMoves[depth] = move;
        searchDepthLimited(maxDepth, depth + 1);
        applyTurn(inverseMoveIndex(move));
    }
}

int main() {
    initCube();

    // —— Hard‑coded scramble ——
    std::vector<std::string> scrambleMoves = { 
        "U", "R2", "F'", "B2", "L"
    };

    if (!scrambleMoves.empty()) {
        std::cout << "Scrambling with:";
        for (auto& token : scrambleMoves) {
            int mv = parseMoveToken(token);
            if (mv < 0) {
                std::cerr << "\nError: invalid scramble token '"
                          << token << "'\n";
                return 1;
            }
            std::cout << " " << moveNotationNames[mv];
            applyTurn(mv);
        }
        std::cout << "\n\n";
    }

    auto startTime = std::chrono::steady_clock::now();
    for (int depth = 1; depth <= MAX_SEARCH_DEPTH; ++depth) {
        searchDepthLimited(depth);
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedMs   = std::chrono::duration_cast<std::chrono::milliseconds>(
                               currentTime - startTime
                           ).count();
        std::cout << "Depth " << depth
                  << " completed in " << elapsedMs << " ms\n";
    }

    return 0;
}
