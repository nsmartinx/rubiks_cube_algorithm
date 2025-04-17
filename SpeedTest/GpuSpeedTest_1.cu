#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>

using u64 = uint64_t;

// —— Configuration ——
#define MAX_SEARCH_DEPTH 8

// —— Cubie‑level raw quarter‑turn moves ——
// corner slots: 0=URF,1=UFL,2=ULB,3=UBR,4=DFR,5=DLF,6=DBL,7=DRB
// edge   slots: 0=UR, 1=UF, 2=UL, 3=UB, 4=DR, 5=DF, 6=DL, 7=DB, 8=FR, 9=FL, 10=BL, 11=BR
struct RawMove {
    int corner_permutation[8];
    int corner_orientation[8];
    int edge_permutation[12];
    int edge_orientation[12];
};

// Host‑side move definitions
RawMove hostQuarterTurns[6] = {
    // U
    { { 3, 0, 1, 2, 4, 5, 6, 7 }, { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    // R
    { { 4, 1, 2, 0, 7, 5, 6, 3 }, { 2, 0, 0, 1, 1, 0, 0, 2 },
      { 8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    // F
    { { 1, 5, 2, 3, 0, 4, 6, 7 }, { 1, 2, 0, 0, 2, 1, 0, 0 },
      { 0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11 }, { 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0 } },
    // D
    { { 0, 1, 2, 3, 5, 6, 7, 4 }, { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    // L
    { { 0, 2, 6, 3, 4, 1, 5, 7 }, { 0, 1, 2, 0, 0, 2, 1, 0 },
      { 0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11 }, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
    // B
    { { 0, 1, 3, 7, 4, 5, 2, 6 }, { 0, 0, 1, 2, 0, 0, 2, 1 },
      { 0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7 }, { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1 } }
};

// Device‑side constant memory for moves
__constant__ RawMove deviceQuarterTurns[6];

// Parse a move token (e.g. "R2" → index 4)
int parseMoveToken(const std::string &token) {
    if (token.empty()) return -1;
    int face;
    switch (token[0]) {
        case 'U': face = 0; break;
        case 'R': face = 1; break;
        case 'F': face = 2; break;
        case 'D': face = 3; break;
        case 'L': face = 4; break;
        case 'B': face = 5; break;
        default: return -1;
    }
    int type = 0;
    if (token.size() == 2) {
        type = (token[1] == '2' ? 1 : 2);
    }
    return face * 3 + type;
}

// Inverse of a move index
inline int inverseMoveIndex(int moveIndex) {
    int faceIndex = moveIndex / 3;
    int turnType  = moveIndex % 3;
    int inverseType = (turnType == 0 ? 2 : (turnType == 1 ? 1 : 0));
    return faceIndex * 3 + inverseType;
}

// Apply one quarter‑turn on CPU
void applyQuarterTurnCPU(u64 &cornerState, u64 &edgeState, int face) {
    RawMove mv = hostQuarterTurns[face];
    u64 oldCorners = cornerState;
    u64 oldEdges   = edgeState;
    u64 newCorners = 0;
    u64 newEdges   = 0;

    for (int i = 0; i < 8; ++i) {
        u64 packed = (oldCorners >> (5 * i)) & 0x1FULL;
        int piece           = packed & 7;
        int orientation     = (packed >> 3) & 3;
        int newOrientation  = (orientation + mv.corner_orientation[i]) % 3;
        int destinationSlot = mv.corner_permutation[i];
        u64 packedOutput    = u64(piece) | (u64(newOrientation) << 3);
        newCorners         |= packedOutput << (5 * destinationSlot);
    }
    for (int j = 0; j < 12; ++j) {
        u64 packed = (oldEdges >> (5 * j)) & 0x1FULL;
        int piece           = packed & 0xF;
        int orientation     = (packed >> 4) & 1;
        int newOrientation  = orientation ^ mv.edge_orientation[j];
        int destinationSlot = mv.edge_permutation[j];
        u64 packedOutput    = u64(piece) | (u64(newOrientation) << 4);
        newEdges           |= packedOutput << (5 * destinationSlot);
    }

    cornerState = newCorners;
    edgeState   = newEdges;
}

// Apply a move (0–17) on CPU
void applyMoveCPU(u64 &cornerState, u64 &edgeState, int moveIndex) {
    int faceIndex = moveIndex / 3;
    int turnType  = moveIndex % 3;
    int repetitions = (turnType == 1 ? 2 : (turnType == 2 ? 3 : 1));
    for (int r = 0; r < repetitions; ++r) {
        applyQuarterTurnCPU(cornerState, edgeState, faceIndex);
    }
}

// GPU quarter‑turn
__device__ void applyQuarterTurn(u64 &cornerState, u64 &edgeState, int face) {
    RawMove mv = deviceQuarterTurns[face];
    u64 oldCorners = cornerState;
    u64 oldEdges   = edgeState;
    u64 newCorners = 0;
    u64 newEdges   = 0;
    for (int i = 0; i < 8; ++i) {
        u64 packed = (oldCorners >> (5 * i)) & 0x1FULL;
        int piece           = packed & 7;
        int orientation     = (packed >> 3) & 3;
        int newOrientation  = (orientation + mv.corner_orientation[i]) % 3;
        int destinationSlot = mv.corner_permutation[i];
        u64 packedOutput    = u64(piece) | (u64(newOrientation) << 3);
        newCorners         |= packedOutput << (5 * destinationSlot);
    }
    for (int j = 0; j < 12; ++j) {
        u64 packed = (oldEdges >> (5 * j)) & 0x1FULL;
        int piece           = packed & 0xF;
        int orientation     = (packed >> 4) & 1;
        int newOrientation  = orientation ^ mv.edge_orientation[j];
        int destinationSlot = mv.edge_permutation[j];
        u64 packedOutput    = u64(piece) | (u64(newOrientation) << 4);
        newEdges           |= packedOutput << (5 * destinationSlot);
    }
    cornerState = newCorners;
    edgeState   = newEdges;
}

// GPU full‑turn
__device__ void applyMove(u64 &cornerState, u64 &edgeState, int moveIndex) {
    int faceIndex = moveIndex / 3;
    int turnType  = moveIndex % 3;
    int repetitions = (turnType == 1 ? 2 : (turnType == 2 ? 3 : 1));
    for (int r = 0; r < repetitions; ++r) {
        applyQuarterTurn(cornerState, edgeState, faceIndex);
    }
}

// Kernel: brute‑force sequences
__global__ void bruteForceKernel(
    u64 scrambledCorners,
    u64 scrambledEdges,
    u64 solvedCorners,
    u64 solvedEdges,
    int depth,
    int *solBuffer,
    int *foundFlag)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalSeq = 1;
    for (int i = 0; i < depth; ++i) totalSeq *= 18ULL;
    if (tid >= totalSeq) return;

    u64 cornerState = scrambledCorners;
    u64 edgeState   = scrambledEdges;
    int seq[MAX_SEARCH_DEPTH];
    unsigned int id = tid;
    for (int i = 0; i < depth; ++i) {
        seq[i] = id % 18;
        id     /= 18;
        applyMove(cornerState, edgeState, seq[i]);
    }
    if (cornerState == solvedCorners && edgeState == solvedEdges) {
        if (atomicExch(foundFlag, 1) == 0) {
            for (int i = 0; i < depth; ++i) solBuffer[i] = seq[i];
        }
    }
}

int main() {
    // Init solved state
    u64 solvedCorners = 0;
    u64 solvedEdges   = 0;
    for (int i = 0; i < 8; ++i) solvedCorners |= (u64(i) << (5 * i));
    for (int j = 0; j < 12; ++j) solvedEdges   |= (u64(j) << (5 * j));

    // Scramble moves
    std::vector<std::string> scrambleMoves = { "U", "R2", "F'", "B2", "L" };
    u64 scrambledCorners = solvedCorners;
    u64 scrambledEdges   = solvedEdges;

    // Log scramble
    std::cout << "Scrambling with:";
    for (const auto &tok : scrambleMoves) std::cout << " " << tok;
    std::cout << std::endl;

    // Apply scramble on CPU
    for (const auto &tok : scrambleMoves) {
        int mv = parseMoveToken(tok);
        if (mv < 0) { std::cerr << "Invalid token: " << tok << std::endl; return 1; }
        applyMoveCPU(scrambledCorners, scrambledEdges, mv);
    }

    // Copy moves to GPU
    cudaMemcpyToSymbol(deviceQuarterTurns, hostQuarterTurns, sizeof(RawMove) * 6);

    // Allocate GPU buffers
    int *d_solution, *d_flag;
    cudaMalloc(&d_solution, sizeof(int) * MAX_SEARCH_DEPTH);
    cudaMalloc(&d_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);

    bool printed = false;
    auto start = std::chrono::steady_clock::now();

    // Depth‑limited search
    for (int depth = 1; depth <= MAX_SEARCH_DEPTH; ++depth) {
        unsigned long long totalSeq = 1;
        for (int i = 0; i < depth; ++i) totalSeq *= 18ULL;
        int tpB = 256;
        int blocks = (totalSeq + tpB - 1ULL) / tpB;
        bruteForceKernel<<<blocks, tpB>>>(
            scrambledCorners,
            scrambledEdges,
            solvedCorners,
            solvedEdges,
            depth,
            d_solution,
            d_flag
        );
        cudaDeviceSynchronize();

        int found;
        cudaMemcpy(&found, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (found && !printed) {
            int sol[MAX_SEARCH_DEPTH];
            cudaMemcpy(sol, d_solution, sizeof(int) * depth, cudaMemcpyDeviceToHost);
            std::cout << "Solution found (" << depth << " moves): ";
            const char *names[18] = { "U","U2","U'","R","R2","R'",
                                     "F","F2","F'","D","D2","D'",
                                     "L","L2","L'","B","B2","B'" };
            for (int i = 0; i < depth; ++i) std::cout << names[sol[i]] << (i+1<depth?' ':'\n');
            printed = true;
        }

        auto now = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        std::cout << "Depth " << depth << " completed in " << ms << " ms" << std::endl;
    }

    cudaFree(d_solution);
    cudaFree(d_flag);
    return 0;
}
