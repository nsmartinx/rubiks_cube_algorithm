#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

using u64 = uint64_t;

// Forward declarations of our four stage solvers.
std::vector<std::string> solveStage1(u64 edgeState);
std::vector<std::string> solveStage2(u64 cornerState, u64 edgeState);
std::vector<std::string> solveStage3(u64 cornerState, u64 edgeState);
std::vector<std::string> solveStage4(u64 cornerState, u64 edgeState);

// —— Cubie‑level raw quarter‑turn moves ——
// corner slots: 0=URF,1=UFL,2=ULB,3=UBR,4=DFR,5=DLF,6=DBL,7=DRB
// edge   slots: 0=UR, 1=UF, 2=UL, 3=UB, 4=DR, 5=DF, 6=DL, 7=DB,
//               8=FR, 9=FL, 10=BL, 11=BR
struct RawMove {
    int  cornerPerm[8];
    int  cornerOrientation[8];
    int  edgePerm[12];
    uint16_t edgeOriMask;   // bit i = flip for edge slot i
};

// —— Fully‑expanded moves ——
RawMove moves18[18] = {
    // U
    {{3,0,1,2,4,5,6,7}, {0,0,0,0,0,0,0,0},
     {3,0,1,2,4,5,6,7,8,9,10,11}, 0b0000'0000'0000},
    // U2
    {{2,3,0,1,4,5,6,7}, {0,0,0,0,0,0,0,0},
     {2,3,0,1,4,5,6,7,8,9,10,11}, 0b0000'0000'0000},
    // U'
    {{1,2,3,0,4,5,6,7}, {0,0,0,0,0,0,0,0},
     {1,2,3,0,4,5,6,7,8,9,10,11}, 0b0000'0000'0000},

    // R
    {{4,1,2,0,7,5,6,3}, {2,0,0,1,1,0,0,2},
     {8,1,2,3,11,5,6,7,4,9,10,0},  0b0000'0000'0000},
    // R2
    {{7,1,2,4,3,5,6,0}, {0,0,0,0,0,0,0,0},
     {4,1,2,3,0,5,6,7,11,9,10,8},  0b0000'0000'0000},
    // R'
    {{3,1,2,7,0,5,6,4}, {2,0,0,1,1,0,0,2},
     {11,1,2,3,8,5,6,7,0,9,10,4},  0b0000'0000'0000},

    // F
    {{1,5,2,3,0,4,6,7}, {1,2,0,0,2,1,0,0},
     {0,9,2,3,4,8,6,7,1,5,10,11},  0b0011'0010'0010},
    // F2
    {{5,4,2,3,1,0,6,7}, {0,0,0,0,0,0,0,0},
     {0,5,2,3,4,1,6,7,9,8,10,11},  0b0000'0000'0000},
    // F'
    {{4,0,2,3,5,1,6,7}, {1,2,0,0,2,1,0,0},
     {0,8,2,3,4,9,6,7,5,1,10,11},  0b0011'0010'0010},

    // D
    {{0,1,2,3,5,6,7,4}, {0,0,0,0,0,0,0,0},
     {0,1,2,3,5,6,7,4,8,9,10,11},  0b0000'0000'0000},
    // D2
    {{0,1,2,3,6,7,4,5}, {0,0,0,0,0,0,0,0},
     {0,1,2,3,6,7,4,5,8,9,10,11},  0b0000'0000'0000},
    // D'
    {{0,1,2,3,7,4,5,6}, {0,0,0,0,0,0,0,0},
     {0,1,2,3,7,4,5,6,8,9,10,11},  0b0000'0000'0000},

    // L
    {{0,2,6,3,4,1,5,7}, {0,1,2,0,0,2,1,0},
     {0,1,10,3,4,5,9,7,8,2,6,11},  0b0000'0000'0000},
    // L2
    {{0,6,5,3,4,2,1,7}, {0,0,0,0,0,0,0,0},
     {0,1,6,3,4,5,2,7,8,10,9,11},  0b0000'0000'0000},
    // L'
    {{0,5,1,3,4,6,2,7}, {0,1,2,0,0,2,1,0},
     {0,1,9,3,4,5,10,7,8,6,2,11},  0b0000'0000'0000},

    // B
    {{0,1,3,7,4,5,2,6}, {0,0,1,2,0,0,2,1},
     {0,1,2,11,4,5,6,10,8,9,3,7},  0b1100'1000'1000},
    // B2
    {{0,1,7,6,4,5,3,2}, {0,0,0,0,0,0,0,0},
     {0,1,2,7,4,5,6,3,8,9,11,10},  0b0000'0000'0000},
    // B'
    {{0,1,6,2,4,5,7,3}, {0,0,1,2,0,0,2,1},
     {0,1,2,10,4,5,6,11,8,9,7,3},  0b1100'1000'1000}
};

const char* moveNotationNames[18] = {
    "U","U2","U'",
    "R","R2","R'",
    "F","F2","F'",
    "D","D2","D'",
    "L","L2","L'",
    "B","B2","B'"
};

// Convert notation token to move index
int parseMoveToken(const std::string& token) {
    if (token.empty()) return -1;
    int face;
    switch (token[0]) {
        case 'U': face = 0; break;
        case 'R': face = 1; break;
        case 'F': face = 2; break;
        case 'D': face = 3; break;
        case 'L': face = 4; break;
        case 'B': face = 5; break;
        default:  return -1;
    }
    int turn = 0;
    if (token.size() == 2) {
        turn = (token[1] == '2' ? 1 : 2);
    }
    return face * 3 + turn;
}

// Apply a raw move on CPU
void applyRawMoveCPU(u64 & cornerState, u64 & edgeState, int moveIndex) {
    const RawMove & m = moves18[moveIndex];
    u64 oldC = cornerState, oldE = edgeState;
    u64 newC = 0,     newE = 0;

    // —— corners ——
    for (int dst = 0; dst < 8; ++dst) {
        int src = m.cornerPerm[dst];
        u64 packed    = (oldC >> (5 * src)) & 0x1F; 
        int pieceIdx  =  packed & 0x7;
        int ori       = (packed >> 3) & 0x3;
        int nori      = (ori + m.cornerOrientation[dst]) % 3;
        u64 outPacked = u64(pieceIdx) | (u64(nori) << 3);
        newC       |= outPacked << (5 * dst);
    }

    // —— edges ——  
    for (int dst = 0; dst < 12; ++dst) {
        int src        = m.edgePerm[dst];
        u64 packed     = (oldE >> (5 * src)) & 0x1F;
        int pieceIdx   = packed & 0xF;
        int ori        = (packed >> 4) & 0x1;
        int nori       = ori ^ ((m.edgeOriMask >> dst) & 1);
        u64 outPacked  = u64(pieceIdx) | (u64(nori) << 4);
        newE          |= outPacked << (5 * dst);
    }

    cornerState = newC;
    edgeState   = newE;
}


int main() {
    // 1) Hardcoded scramble:
    std::vector<std::string> scramble = {
        "R2","D'","B'","R","B'","U2","B'","U'",
        "L'","U'","F'","L2","B2","F'","L2","B'",
        "U","B","F2","L2","B2","F'","L2","R'","U'"
    };

    // 2) Initialize to solved cube
    u64 cornerState = 0, edgeState = 0;
    for (int i = 0; i < 8; ++i) cornerState |= (u64(i) << (5*i));
    for (int i = 0; i < 12; ++i) edgeState |= (u64(i) << (5*i));

    // 3) Apply scramble on CPU
    std::cout << "Scrambling with:";
    for (auto& mv : scramble) {
        int idx = parseMoveToken(mv);
        if (idx < 0) {
            std::cerr << "\nError: invalid move '" << mv << "'\n";
            return 1;
        }
        std::cout << " " << mv;
        applyRawMoveCPU(cornerState, edgeState, idx);
    }
    std::cout << "\n\n";

    // 4‑stage solver
    std::vector<std::string> totalSolution;

    // Stage 1
    auto st1 = solveStage1(edgeState);
    std::cout << "Stage 1 solution:";
    for (auto& mv : st1) {
        std::cout << " " << mv;
        totalSolution.push_back(mv);
        applyRawMoveCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout << "\nLength: " << st1.size() << "\n";
    std::cout << "\n\n";

    // Stage 2
    auto st2 = solveStage2(cornerState, edgeState);
    std::cout << "Stage 2 solution:";
    for (auto& mv : st2) {
        std::cout << " " << mv;
        totalSolution.push_back(mv);
        applyRawMoveCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout << "\nLength: " << st2.size() << "\n";
    std::cout << "\n\n";

    // Stage 3
    auto st3 = solveStage3(cornerState, edgeState);
    std::cout << "Stage 3 solution:";
    for (auto& mv : st3) {
        std::cout << " " << mv;
        totalSolution.push_back(mv);
        applyRawMoveCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout << "\nLength: " << st3.size() << "\n";
    std::cout << "\n\n";

    // Stage 4
    auto st4 = solveStage4(cornerState, edgeState);
    //std::vector<std::string> st4 = {};
    std::cout << "Stage 4 solution:";
    for (auto& mv : st4) {
        std::cout << " " << mv;
        totalSolution.push_back(mv);
        applyRawMoveCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout << "\nLength: " << st4.size() << "\n";
    std::cout << "\n\n";

    // Print the complete solution
    std::cout << "Full solution:";
    for (auto& mv : totalSolution) {
        std::cout << " " << mv;
    }
    std::cout << "\nLength: " << totalSolution.size() << "\n";
    std::cout << "\n";

    return 0;

}
