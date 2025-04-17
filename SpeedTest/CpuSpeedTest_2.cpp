#include <iostream>
#include <chrono>
#include <cstdint>
#include <vector>
#include <string>

using u64 = uint64_t;

// —— Configuration ——
// maximum brute‑force depth
#define MAX_SEARCH_DEPTH 7

// —— Cubie‑level raw quarter‑turn moves ——
// corner slots: 0=URF,1=UFL,2=ULB,3=UBR,4=DFR,5=DLF,6=DBL,7=DRB
// edge   slots: 0=UR, 1=UF, 2=UL, 3=UB, 4=DR, 5=DF, 6=DL, 7=DB, 8=FR, 9=FL, 10=BL, 11=BR
struct RawMove {
    int cornerPerm[8];
    int cornerOrientation[8];
    int edgePerm[12];
    int edgeOrientation[12];
};

// Precomputed raw moves for all 18 face turns
RawMove moves[18] = {
    { {3,0,1,2,4,5,6,7}, {0,0,0,0,0,0,0,0}, {3,0,1,2,4,5,6,7,8,9,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // U
    { {2,3,0,1,4,5,6,7}, {0,0,0,0,0,0,0,0}, {2,3,0,1,4,5,6,7,8,9,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // U2
    { {1,2,3,0,4,5,6,7}, {0,0,0,0,0,0,0,0}, {1,2,3,0,4,5,6,7,8,9,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // U'
    { {4,1,2,0,7,5,6,3}, {2,0,0,1,1,0,0,2}, {8,1,2,3,11,5,6,7,4,9,10,0}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // R
    { {7,1,2,4,3,5,6,0}, {0,0,0,0,0,0,0,0}, {4,1,2,3,0,5,6,7,11,9,10,8}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // R2
    { {3,1,2,7,0,5,6,4}, {2,0,0,1,1,0,0,2}, {11,1,2,3,8,5,6,7,0,9,10,4}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // R'
    { {1,5,2,3,0,4,6,7}, {1,2,0,0,2,1,0,0}, {0,9,2,3,4,8,6,7,1,5,10,11}, {0,1,0,0,0,1,0,0,1,1,0,0} }, // F
    { {5,4,2,3,1,0,6,7}, {0,0,0,0,0,0,0,0}, {0,5,2,3,4,1,6,7,9,8,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // F2
    { {4,0,2,3,5,1,6,7}, {1,2,0,0,2,1,0,0}, {0,8,2,3,4,9,6,7,5,1,10,11}, {0,1,0,0,0,1,0,0,1,1,0,0} }, // F'
    { {0,1,2,3,5,6,7,4}, {0,0,0,0,0,0,0,0}, {0,1,2,3,5,6,7,4,8,9,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // D
    { {0,1,2,3,6,7,4,5}, {0,0,0,0,0,0,0,0}, {0,1,2,3,6,7,4,5,8,9,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // D2
    { {0,1,2,3,7,4,5,6}, {0,0,0,0,0,0,0,0}, {0,1,2,3,7,4,5,6,8,9,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // D'
    { {0,2,6,3,4,1,5,7}, {0,1,2,0,0,2,1,0}, {0,1,10,3,4,5,9,7,8,2,6,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // L
    { {0,6,5,3,4,2,1,7}, {0,0,0,0,0,0,0,0}, {0,1,6,3,4,5,2,7,8,10,9,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // L2
    { {0,5,1,3,4,6,2,7}, {0,1,2,0,0,2,1,0}, {0,1,9,3,4,5,10,7,8,6,2,11}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // L'
    { {0,1,3,7,4,5,2,6}, {0,0,1,2,0,0,2,1}, {0,1,2,11,4,5,6,10,8,9,3,7}, {0,0,0,1,0,0,0,1,0,0,1,1} }, // B
    { {0,1,7,6,4,5,3,2}, {0,0,0,0,0,0,0,0}, {0,1,2,7,4,5,6,3,8,9,11,10}, {0,0,0,0,0,0,0,0,0,0,0,0} }, // B2
    { {0,1,6,2,4,5,7,3}, {0,0,1,2,0,0,2,1}, {0,1,2,10,4,5,6,11,8,9,7,3}, {0,0,0,1,0,0,0,1,0,0,1,1} }  // B'
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

// record current path
int solutionMoves[MAX_SEARCH_DEPTH];

// apply any of the 18 moves directly
void applyTurn(int move) {
    const RawMove& mv = moves[move];
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

// initialize to solved state
void initCube() {
    cornersState = edgesState = 0;
    for (int i = 0; i < 8; ++i) cornersState |= (u64(i) << (5 * i));
    for (int j = 0; j < 12; ++j) edgesState   |= (u64(j) << (5 * j));
    solvedCornersState = cornersState;
    solvedEdgesState   = edgesState;
}

// print the found solution
void printSolution(int depth) {
    std::cout << "Solution found (" << depth << " moves): ";
    for (int i = 0; i < depth; ++i) {
        std::cout << moveNotationNames[solutionMoves[i]]
                  << (i + 1 < depth ? " " : "");
    }
    std::cout << "\n";
}

// depth‑limited brute force, prevents same-face consecutive moves
void searchDepthLimited(int maxDepth, int depth = 0) {
    if (!foundSolution && depth > 0
        && cornersState == solvedCornersState
        && edgesState   == solvedEdgesState) {
        printSolution(depth);
        foundSolution = true;
    }
    if (depth == maxDepth) return;

    for (int move = 0; move < 18; ++move) {
        // skip if same face as previous move
        if (depth > 0 && (move / 3) == (solutionMoves[depth - 1] / 3)) continue;
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
