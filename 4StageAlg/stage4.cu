#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using u64 = uint64_t;
using u32 = uint32_t;

#define STAGE4_MAX_DEPTH 15

// Device constant: allowed double-turn moves (indices into 18-move table)
__device__ __constant__ int d_allowedDoubleTurnMoves[6] = {
    1, 4, 7, 10, 13, 16  // U2, R2, F2, D2, L2, B2
};

// Device constant: solved-state labels for comparison
__device__ __constant__ u32 d_identityCornerPermutation = 0x00FAC688u;
__device__ __constant__ u64 d_identityEdgePermutation   = 0x0BA9876543210ULL;

// Device constant: corner permutation for each move (dest to src)
// 3 arrays per row, 6 rows, comments at end with move names
__device__ __constant__ int d_cornerPermutation[18][8] = {
    {3, 0, 1, 2, 4, 5, 6, 7}, {2, 3, 0, 1, 4, 5, 6, 7}, {1, 2, 3, 0, 4, 5, 6, 7}, // U, U2, U'
    {4, 1, 2, 0, 7, 5, 6, 3}, {7, 1, 2, 4, 3, 5, 6, 0}, {3, 1, 2, 7, 0, 5, 6, 4}, // R, R2, R'
    {1, 5, 2, 3, 0, 4, 6, 7}, {5, 4, 2, 3, 1, 0, 6, 7}, {4, 0, 2, 3, 5, 1, 6, 7}, // F, F2, F'
    {0, 1, 2, 3, 5, 6, 7, 4}, {0, 1, 2, 3, 6, 7, 4, 5}, {0, 1, 2, 3, 7, 4, 5, 6}, // D, D2, D'
    {0, 2, 6, 3, 4, 1, 5, 7}, {0, 6, 5, 3, 4, 2, 1, 7}, {0, 5, 1, 3, 4, 6, 2, 7}, // L, L2, L'
    {0, 1, 3, 7, 4, 5, 2, 6}, {0, 1, 7, 6, 4, 5, 3, 2}, {0, 1, 6, 2, 4, 5, 7, 3}  // B, B2, B'
};

// Device constant: edge permutation for each move (dest to src)
// 3 arrays per row, 6 rows, comments at end with move names
__device__ __constant__ int d_edgePermutation[18][12] = {
    {3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11}, {2, 3, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11}, {1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11}, // U, U2, U'
    {8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0}, {4, 1, 2, 3, 0, 5, 6, 7, 11, 9, 10, 8}, {11, 1, 2, 3, 8, 5, 6, 7, 0, 9, 10, 4}, // R, R2, R'
    {0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11}, {0, 5, 2, 3, 4, 1, 6, 7, 9, 8, 10, 11}, {0, 8, 2, 3, 4, 9, 6, 7, 5, 1, 10, 11}, // F, F2, F'
    {0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11}, {0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11}, {0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11}, // D, D2, D'
    {0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11}, {0, 1, 6, 3, 4, 5, 2, 7, 8, 10, 9, 11}, {0, 1, 9, 3, 4, 5, 10, 7, 8, 6, 2, 11}, // L, L2, L'
    {0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7}, {0, 1, 2, 7, 4, 5, 6, 3, 8, 9, 11, 10}, {0, 1, 2, 10, 4, 5, 6, 11, 8, 9, 7, 3}  // B, B2, B'
};

// Apply corner-permutation move (ignore orientation)
__device__ u32 applyCornerPermutationMoveGpu2(u32 state, int moveIndex) {
    u32 result = 0;
    for (int dest = 0; dest < 8; ++dest) {
        int src = d_cornerPermutation[moveIndex][dest];
        u32 piece = (state >> (3 * src)) & 7u;
        result |= piece << (3 * dest);
    }
    return result;
}

// Apply edge-permutation move (4 bits per edge)
__device__ u64 applyEdgePermutationMoveGpu(u64 state, int moveIndex) {
    u64 result = 0;
    for (int dest = 0; dest < 12; ++dest) {
        int src = d_edgePermutation[moveIndex][dest];
        u64 piece = (state >> (4 * src)) & 0xFull;
        result |= piece << (4 * dest);
    }
    return result;
}

// Kernel: brute-force search for stage 4
__global__ void bruteForceStage4Kernel(
    u32 startCornerState,
    u64 startEdgeState,
    int searchDepth,
    int *d_solutionMoveBuffer,
    int *d_solutionFoundFlag
) {
    unsigned long long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalSequences = 1ULL;
    for (int i = 0; i < searchDepth; ++i) totalSequences *= 6ULL;
    if (threadIndex >= totalSequences) return;

    u32 cornerState = startCornerState;
    u64 edgeState = startEdgeState;
    int localMoveSequence[STAGE4_MAX_DEPTH];
    unsigned long long sequenceCode = threadIndex;
    int previousFaceIndex = -1;

    for (int step = 0; step < searchDepth; ++step) {
        int selectionIndex = sequenceCode % 6;
        sequenceCode /= 6;
        int moveIndex = d_allowedDoubleTurnMoves[selectionIndex];
        int faceIndex = moveIndex / 3;
        if (faceIndex == previousFaceIndex) return;
        previousFaceIndex = faceIndex;

        localMoveSequence[step] = moveIndex;
        cornerState = applyCornerPermutationMoveGpu2(cornerState, moveIndex);
        edgeState = applyEdgePermutationMoveGpu(edgeState, moveIndex);
    }

    if (cornerState == d_identityCornerPermutation &&
        edgeState == d_identityEdgePermutation)
    {
        if (atomicExch(d_solutionFoundFlag, 1) == 0) {
            for (int i = 0; i < searchDepth; ++i) {
                d_solutionMoveBuffer[i] = localMoveSequence[i];
            }
        }
    }
}

// Pack raw corner labels into 3 bits per slot
static u32 packCornerState(u64 rawCornerState) {
    u32 result = 0;
    for (int i = 0; i < 8; ++i) {
        u32 piece = (rawCornerState >> (5 * i)) & 7u;
        result |= piece << (3 * i);
    }
    return result;
}

// Pack raw edge labels into 4 bits per slot
static u64 packEdgeState(u64 rawEdgeState) {
    u64 result = 0;
    for (int i = 0; i < 12; ++i) {
        u64 piece = (rawEdgeState >> (5 * i)) & 0x1Full;
        result |= piece << (4 * i);
    }
    return result;
}

std::vector<std::string> solveStage4(u64 rawCornerState, u64 rawEdgeState) {
    u32 h_startCornerState = packCornerState(rawCornerState);
    u64 h_startEdgeState = packEdgeState(rawEdgeState);

    int *d_solutionMoveBuffer;
    int *d_solutionFoundFlag;
    cudaMalloc(&d_solutionMoveBuffer, sizeof(int) * STAGE4_MAX_DEPTH);
    cudaMalloc(&d_solutionFoundFlag, sizeof(int));
    int h_initialFlag = 0;
    cudaMemcpy(d_solutionFoundFlag, &h_initialFlag, sizeof(h_initialFlag), cudaMemcpyHostToDevice);

    int h_solutionMoveIndices[STAGE4_MAX_DEPTH] = {0};
    int h_solutionDepth = 0;
    auto h_searchStartTime = std::chrono::steady_clock::now();

    for (int depth = 1; depth <= STAGE4_MAX_DEPTH; ++depth) {
        unsigned long long totalSequences = 1ULL;
        for (int i = 0; i < depth; ++i) totalSequences *= 6ULL;
        int threadsPerBlock = 256;
        int blockCount = (totalSequences + threadsPerBlock - 1ULL) / threadsPerBlock;

        bruteForceStage4Kernel<<<blockCount, threadsPerBlock>>>(
            h_startCornerState,
            h_startEdgeState,
            depth,
            d_solutionMoveBuffer,
            d_solutionFoundFlag
        );
        cudaDeviceSynchronize();

        int h_foundFlag = 0;
        cudaMemcpy(&h_foundFlag, d_solutionFoundFlag, sizeof(h_foundFlag), cudaMemcpyDeviceToHost);
        if (h_foundFlag) {
            h_solutionDepth = depth;
            cudaMemcpy(h_solutionMoveIndices, d_solutionMoveBuffer, sizeof(int) * depth, cudaMemcpyDeviceToHost);
            break;
        }
    }

    auto h_searchEndTime = std::chrono::steady_clock::now();
    auto h_elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(h_searchEndTime - h_searchStartTime).count();
    std::cout << "Stage 4 (Solved) solved in " << h_elapsedMs << " ms\n";

    cudaFree(d_solutionMoveBuffer);
    cudaFree(d_solutionFoundFlag);

    static const char* h_moveNames[18] = {
        "U","U2","U'","R","R2","R'",
        "F","F2","F'","D","D2","D'",
        "L","L2","L'","B","B2","B'"
    };
    std::vector<std::string> h_solutionMoves;
    h_solutionMoves.reserve(h_solutionDepth);
    for (int i = 0; i < h_solutionDepth; ++i) {
        h_solutionMoves.emplace_back(h_moveNames[h_solutionMoveIndices[i]]);
    }
    return h_solutionMoves;
}
