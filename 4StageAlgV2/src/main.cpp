#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <sstream>
#include "common.h"

// Globals for the GPU packers in solvers.cu
u64 g_rawEdgeState;
u64 g_rawCornerState;

// Forward declarations of the GPU‐driven solvers
std::vector<std::string> solveStage1();
std::vector<std::string> solveStage2();
std::vector<std::string> solveStage3();
std::vector<std::string> solveStage4();

// Raw quarter‐turn moves at the cubie level
struct RawMove {
    int        cornerPermutation[8];     // which piece lands in slot i
    int        cornerOrientation[8];     // how that piece’s twist changes
    int        edgePermutation[12];      // which piece lands in slot i
    uint16_t   edgeOrientation;          // bitmask of which edges flip
};

RawMove rawMoves[18] = {
    { {3,0,1,2,4,5,6,7}, {0,0,0,0,0,0,0,0},
      {3,0,1,2,4,5,6,7,8,9,10,11}, 0b0000'0000'0000 }, // U
    { {2,3,0,1,4,5,6,7}, {0,0,0,0,0,0,0,0},
      {2,3,0,1,4,5,6,7,8,9,10,11}, 0b0000'0000'0000 }, // U2
    { {1,2,3,0,4,5,6,7}, {0,0,0,0,0,0,0,0},
      {1,2,3,0,4,5,6,7,8,9,10,11}, 0b0000'0000'0000 }, // U'

    { {4,1,2,0,7,5,6,3}, {2,0,0,1,1,0,0,2},
      {8,1,2,3,11,5,6,7,4,9,10,0}, 0b0000'0000'0000 }, // R
    { {7,1,2,4,3,5,6,0}, {0,0,0,0,0,0,0,0},
      {4,1,2,3,0,5,6,7,11,9,10,8}, 0b0000'0000'0000 }, // R2
    { {3,1,2,7,0,5,6,4}, {2,0,0,1,1,0,0,2},
      {11,1,2,3,8,5,6,7,0,9,10,4}, 0b0000'0000'0000 }, // R'

    { {1,5,2,3,0,4,6,7}, {1,2,0,0,2,1,0,0},
      {0,9,2,3,4,8,6,7,1,5,10,11}, 0b0011'0010'0010 }, // F
    { {5,4,2,3,1,0,6,7}, {0,0,0,0,0,0,0,0},
      {0,5,2,3,4,1,6,7,9,8,10,11}, 0b0000'0000'0000 }, // F2
    { {4,0,2,3,5,1,6,7}, {1,2,0,0,2,1,0,0},
      {0,8,2,3,4,9,6,7,5,1,10,11}, 0b0011'0010'0010 }, // F'

    { {0,1,2,3,5,6,7,4}, {0,0,0,0,0,0,0,0},
      {0,1,2,3,5,6,7,4,8,9,10,11}, 0b0000'0000'0000 }, // D
    { {0,1,2,3,6,7,4,5}, {0,0,0,0,0,0,0,0},
      {0,1,2,3,6,7,4,5,8,9,10,11}, 0b0000'0000'0000 }, // D2
    { {0,1,2,3,7,4,5,6}, {0,0,0,0,0,0,0,0},
      {0,1,2,3,7,4,5,6,8,9,10,11}, 0b0000'0000'0000 }, // D'

    { {0,2,6,3,4,1,5,7}, {0,1,2,0,0,2,1,0},
      {0,1,10,3,4,5,9,7,8,2,6,11}, 0b0000'0000'0000 }, // L
    { {0,6,5,3,4,2,1,7}, {0,0,0,0,0,0,0,0},
      {0,1,6,3,4,5,2,7,8,10,9,11}, 0b0000'0000'0000 }, // L2
    { {0,5,1,3,4,6,2,7}, {0,1,2,0,0,2,1,0},
      {0,1,9,3,4,5,10,7,8,6,2,11}, 0b0000'0000'0000 }, // L'

    { {0,1,3,7,4,5,2,6}, {0,0,1,2,0,0,2,1},
      {0,1,2,11,4,5,6,10,8,9,3,7}, 0b1100'1000'1000 }, // B
    { {0,1,7,6,4,5,3,2}, {0,0,0,0,0,0,0,0},
      {0,1,2,7,4,5,6,3,8,9,11,10}, 0b0000'0000'0000 }, // B2
    { {0,1,6,2,4,5,7,3}, {0,0,1,2,0,0,2,1},
      {0,1,2,10,4,5,6,11,8,9,7,3}, 0b1100'1000'1000 }  // B'
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
    if (token.empty()) return -1;
    int face = std::string("URFDLB").find(token[0]);
    if (face == std::string::npos) return -1;
    int offset = 0;
    if (token.size()==2) offset = (token[1]=='2' ? 1 : 2);
    return face*3 + offset;
}

void applyRawMoveOnCPU(u64& cornerState, u64& edgeState, int m) {
    const RawMove& mv = rawMoves[m];
    u64 oldC = cornerState, oldE = edgeState;
    u64 newC = 0, newE = 0;
    // corners
    for (int d=0; d<8; ++d) {
        int s = mv.cornerPermutation[d];
        u64 p = (oldC>>(5*s)) & 0x1F;
        int piece = p & 7, ori = (p>>3)&3;
        int no = (ori + mv.cornerOrientation[d]) % 3;
        u64 op = u64(piece) | (u64(no)<<3);
        newC |= op << (5*d);
    }
    // edges
    for (int d=0; d<12; ++d) {
        int s = mv.edgePermutation[d];
        u64 p = (oldE>>(5*s)) & 0x1F;
        int piece = p & 0xF, ori = (p>>4)&1;
        int no = ori ^ ((mv.edgeOrientation>>d)&1);
        u64 op = u64(piece) | (u64(no)<<4);
        newE |= op << (5*d);
    }
    cornerState = newC;
    edgeState   = newE;
}

void flushAccumulatedMoves(std::vector<std::string>& out, char& lastFace, int& acc) {
    if (!lastFace) return;
    int r = acc % 4; if (r<0) r+=4;
    if (r==1)      out.push_back({lastFace});
    else if (r==2) out.push_back(std::string(1,lastFace)+"2");
    else if (r==3) out.push_back(std::string(1,lastFace)+"'");
    lastFace=0; acc=0;
}

std::vector<std::string> splitString(const std::string& in) {
    std::vector<std::string> tok;
    std::istringstream ss(in);
    std::string t;
    while (ss>>t) tok.push_back(t);
    return tok;
}

int main(){
    // Example scramble
    std::vector<std::string> scramble = {
      "R2","D'","B'","R","B'","U2","B'","U'","L'",
      "U'","F'","L2","B2","F'","L2","B'","U","B",
      "F2","L2","B2","F'","L2","R'","U'"
    };

    // Start in solved state
    u64 cornerState = 0, edgeState = 0;
    for (int i=0; i<8; ++i) cornerState |= u64(i)   << (5*i);
    for (int i=0; i<12;++i) edgeState   |= u64(i)   << (5*i);

    // Apply scramble
    std::cout<<"Scrambling with:";
    for (auto& mv : scramble) {
        int idx = parseMoveToken(mv);
        std::cout<<" "<<mv;
        applyRawMoveOnCPU(cornerState, edgeState, idx);
    }
    std::cout<<"\n\n";

    // Store into globals for GPU packers
    g_rawCornerState = cornerState;
    g_rawEdgeState   = edgeState;

    std::vector<std::string> totalSolution;

    // Stage 1
    auto sol1 = solveStage1();
    std::cout<<"Stage 1:";
    for (auto& mv : sol1) {
        std::cout<<" "<<mv;
        totalSolution.push_back(mv);
        applyRawMoveOnCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout<<"\nLength: "<<sol1.size()<<"\n\n";
    g_rawCornerState = cornerState;
    g_rawEdgeState   = edgeState;

    // Stage 2
    auto sol2 = solveStage2();
    std::cout<<"Stage 2:";
    for (auto& mv : sol2) {
        std::cout<<" "<<mv;
        totalSolution.push_back(mv);
        applyRawMoveOnCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout<<"\nLength: "<<sol2.size()<<"\n\n";
    g_rawCornerState = cornerState;
    g_rawEdgeState   = edgeState;

    // Stage 3
    auto sol3 = solveStage3();
    std::cout<<"Stage 3:";
    for (auto& mv : sol3) {
        std::cout<<" "<<mv;
        totalSolution.push_back(mv);
        applyRawMoveOnCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout<<"\nLength: "<<sol3.size()<<"\n\n";
    g_rawCornerState = cornerState;
    g_rawEdgeState   = edgeState;

    // Stage 4
    auto sol4 = solveStage4();
    std::cout<<"Stage 4:";
    for (auto& mv : sol4) {
        std::cout<<" "<<mv;
        totalSolution.push_back(mv);
        applyRawMoveOnCPU(cornerState, edgeState, parseMoveToken(mv));
    }
    std::cout<<"\nLength: "<<sol4.size()<<"\n\n";

    // Simplify full solution
    std::vector<std::string> simplified;
    char lastFace = 0; int acc = 0;
    for (auto& mv : totalSolution) {
        char f = mv[0];
        int turns = (mv.size()==2 ? (mv[1]=='2'?2:3) : 1);
        if (f != lastFace) {
            flushAccumulatedMoves(simplified,lastFace,acc);
            lastFace = f;
            acc = turns;
        } else {
            acc = (acc + turns)%4;
        }
    }
    flushAccumulatedMoves(simplified,lastFace,acc);

    std::cout<<"Full solution:";
    for (auto& mv : simplified) std::cout<<" "<<mv;
    std::cout<<"\nLength: "<<simplified.size()<<"\n";

    return 0;
}
