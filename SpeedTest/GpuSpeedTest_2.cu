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
    int cornerPerm[8];
    int cornerOrientation[8];
    int edgePerm[12];
    int edgeOrientation[12];
};

// Precomputed raw moves for all 18 face turns
RawMove moves18[18] = {
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

const char * moveNotationNames [18] = {
    "U", "U2", "U'",
    "R", "R2", "R'",
    "F", "F2", "F'",
    "D", "D2", "D'",
    "L", "L2", "L'",
    "B", "B2", "B'"
};

// Device‑side constant memory for moves
__constant__ RawMove deviceMoves[18];

// Convert notation token to move index (0–17)
int parseMoveToken(const std::string & token) {
    if (token.empty()) {
        return -1;
    }
    int faceIndex;
    switch (token[0]) {
        case 'U': faceIndex = 0; break;
        case 'R': faceIndex = 1; break;
        case 'F': faceIndex = 2; break;
        case 'D': faceIndex = 3; break;
        case 'L': faceIndex = 4; break;
        case 'B': faceIndex = 5; break;
        default:
            return -1;
    }
    int turnType = 0;
    if (token.size() == 2) {
        turnType = (token[1] == '2') ? 1 : 2;
    }
    return faceIndex * 3 + turnType;
}

// Apply a raw move on CPU
void applyRawMoveCPU(u64 & cornerState,
                     u64 & edgeState,
                     int moveIndex)
{
    RawMove & move = moves18[moveIndex];
    u64 oldCornerState = cornerState;
    u64 oldEdgeState   = edgeState;
    u64 newCornerState = 0;
    u64 newEdgeState   = 0;

    // Update corners
    for (int slot = 0; slot < 8; ++slot) {
        u64 packed = (oldCornerState >> (5 * slot)) & 0x1FULL;
        int pieceIndex      = packed & 0x7;
        int orientation     = (packed >> 3) & 0x3;
        int newOrientation  = (orientation
                    + move.cornerOrientation[slot]) % 3;
        int destinationSlot = move.cornerPerm[slot];
        u64 outPacked = u64(pieceIndex)
                      | (u64(newOrientation) << 3);
        newCornerState |= outPacked
                        << (5 * destinationSlot);
    }

    // Update edges
    for (int slot = 0; slot < 12; ++slot) {
        u64 packed = (oldEdgeState >> (5 * slot)) & 0x1FULL;
        int pieceIndex      = packed & 0xF;
        int orientation     = (packed >> 4) & 0x1;
        int newOrientation  = orientation ^ move.edgeOrientation[slot];
        int destinationSlot = move.edgePerm[slot];
        u64 outPacked = u64(pieceIndex)
                      | (u64(newOrientation) << 4);
        newEdgeState |= outPacked
                      << (5 * destinationSlot);
    }

    cornerState = newCornerState;
    edgeState   = newEdgeState;
}

// Apply a raw move on GPU
__device__ void applyRawMoveGPU(u64 & cornerState,
                                u64 & edgeState,
                                int moveIndex)
{
    RawMove move = deviceMoves[moveIndex];
    u64 oldCornerState = cornerState;
    u64 oldEdgeState   = edgeState;
    u64 newCornerState = 0;
    u64 newEdgeState   = 0;

    for (int slot = 0; slot < 8; ++slot) {
        u64 packed = (oldCornerState >> (5 * slot)) & 0x1FULL;
        int pieceIndex      = packed & 0x7;
        int orientation     = (packed >> 3) & 0x3;
        int newOrientation  = (orientation
                    + move.cornerOrientation[slot]) % 3;
        int destinationSlot = move.cornerPerm[slot];
        u64 outPacked = u64(pieceIndex)
                      | (u64(newOrientation) << 3);
        newCornerState |= outPacked
                        << (5 * destinationSlot);
    }

    for (int slot = 0; slot < 12; ++slot) {
        u64 packed = (oldEdgeState >> (5 * slot)) & 0x1FULL;
        int pieceIndex      = packed & 0xF;
        int orientation     = (packed >> 4) & 0x1;
        int newOrientation  = orientation ^ move.edgeOrientation[slot];
        int destinationSlot = move.edgePerm[slot];
        u64 outPacked       = u64(pieceIndex) | (u64(newOrientation) << 4);
        newEdgeState       |= outPacked << (5 * destinationSlot);
    }

    cornerState = newCornerState;
    edgeState   = newEdgeState;
}

// Kernel: brute‑force each sequence, skipping consecutive same-face moves
__global__ void bruteForceKernel(
    u64 scrambledCornerState,
    u64 scrambledEdgeState,
    u64 solvedCornerState,
    u64 solvedEdgeState,
    int searchDepth,
    int * solutionBuffer,
    int * solutionFoundFlag)
{
    unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalSequences = 1;
    for (int i = 0; i < searchDepth; ++i) {
        totalSequences *= 18ULL;
    }
    if (threadId >= totalSequences) {
        return;
    }

    u64 cornerState = scrambledCornerState;
    u64 edgeState   = scrambledEdgeState;
    int moveSequence[MAX_SEARCH_DEPTH];
    unsigned int idCopy = threadId;

    for (int i = 0; i < searchDepth; ++i) {
        moveSequence[i] = idCopy % 18;
        idCopy         /= 18;
        // skip if same face as previous
        if (i > 0
            && moveSequence[i] / 3
               == moveSequence[i - 1] / 3) {
            return;
        }
        applyRawMoveGPU(
            cornerState,
            edgeState,
            moveSequence[i]
        );
    }

    if (cornerState == solvedCornerState
        && edgeState   == solvedEdgeState) {
        if (atomicExch(solutionFoundFlag, 1) == 0) {
            for (int i = 0; i < searchDepth; ++i) {
                solutionBuffer[i] = moveSequence[i];
            }
        }
    }
}

int main() {
    // Initialize solved state
    u64 solvedCornerState = 0;
    u64 solvedEdgeState   = 0;
    for (int i = 0; i < 8; ++i) {
        solvedCornerState |= (u64(i) << (5 * i));
    }
    for (int j = 0; j < 12; ++j) {
        solvedEdgeState   |= (u64(j) << (5 * j));
    }

    // Hard‑coded scramble
    std::vector<std::string> scrambleMoves = {
        "U", "R2", "F'", "B2", "L"
    };
    u64 scrambledCornerState = solvedCornerState;
    u64 scrambledEdgeState   = solvedEdgeState;

    std::cout << "Scrambling with:";
    for (const auto & token : scrambleMoves) {
        std::cout << " " << token;
        int moveIndex = parseMoveToken(token);
        if (moveIndex < 0) {
            std::cerr << "Error: invalid token '"
                      << token << "'\n";
            return 1;
        }
        applyRawMoveCPU(
            scrambledCornerState,
            scrambledEdgeState,
            moveIndex
        );
    }
    std::cout << std::endl;

    // Copy moves to GPU constant memory
    cudaMemcpyToSymbol(
        deviceMoves,
        moves18,
        sizeof(RawMove) * 18
    );

    // Allocate device buffers
    int * d_solution;
    int * d_solutionFlag;
    cudaMalloc(& d_solution,
               sizeof(int) * MAX_SEARCH_DEPTH);
    cudaMalloc(& d_solutionFlag,
               sizeof(int));
    int zero = 0;
    cudaMemcpy(
        d_solutionFlag,
        & zero,
        sizeof(int),
        cudaMemcpyHostToDevice
    );

    bool solutionPrinted = false;
    auto startTime = std::chrono::steady_clock::now();

    for (int depth = 1; depth <= MAX_SEARCH_DEPTH; ++depth) {
        unsigned long long totalSequences = 1;
        for (int i = 0; i < depth; ++i) {
            totalSequences *= 18ULL;
        }
        int threadsPerBlock = 256;
        int blockCount      = static_cast<int>(
            (totalSequences + threadsPerBlock - 1)
            / threadsPerBlock
        );

        bruteForceKernel<<<
            blockCount,
            threadsPerBlock
        >>>(
            scrambledCornerState,
            scrambledEdgeState,
            solvedCornerState,
            solvedEdgeState,
            depth,
            d_solution,
            d_solutionFlag
        );
        cudaDeviceSynchronize();

        int found;
        cudaMemcpy(
            & found,
            d_solutionFlag,
            sizeof(int),
            cudaMemcpyDeviceToHost
        );

        if (found && ! solutionPrinted) {
            int solutionMoves[MAX_SEARCH_DEPTH];
            cudaMemcpy(
                solutionMoves,
                d_solution,
                sizeof(int) * depth,
                cudaMemcpyDeviceToHost
            );
            std::cout << "Solution found ("
                      << depth << " moves): ";
            for (int i = 0; i < depth; ++i) {
                std::cout << moveNotationNames[
                    solutionMoves[i]
                ]
                << (i + 1 < depth ? " " : "\n");
            }
            solutionPrinted = true;
        }

        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedMs   = std::chrono::duration_cast<
            std::chrono::milliseconds
        >(
            currentTime - startTime
        ).count();
        std::cout << "Depth " << depth
                  << " completed in "
                  << elapsedMs << " ms" << std::endl;
    }

    cudaFree(d_solution);
    cudaFree(d_solutionFlag);
    return 0;
}
