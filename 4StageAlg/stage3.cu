#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;

#define STAGE3_MAX_DEPTH 13

// Device constant: allowed Thistlethwaite G2 moves
__device__ __constant__ int deviceAllowedMoves[10] = {
    0,1,2,   // U, U2, U'
    4,       // R2
    9,10,11, // D, D2, D'
    13,      // L2
    7,       // F2
    16       // B2
};

// Device constant: corner permutation for each move (dest to src)
__device__ __constant__ int cornerPermutation[18][8] = {
    {3,0,1,2,4,5,6,7}, {2,3,0,1,4,5,6,7}, {1,2,3,0,4,5,6,7}, // U,U2,U'
    {4,1,2,0,7,5,6,3}, {7,1,2,4,3,5,6,0}, {3,1,2,7,0,5,6,4}, // R,R2,R'
    {1,5,2,3,0,4,6,7}, {5,4,2,3,1,0,6,7}, {4,0,2,3,5,1,6,7}, // F,F2,F'
    {0,1,2,3,5,6,7,4}, {0,1,2,3,6,7,4,5}, {0,1,2,3,7,4,5,6}, // D,D2,D'
    {0,2,6,3,4,1,5,7}, {0,6,5,3,4,2,1,7}, {0,5,1,3,4,6,2,7}, // L,L2,L'
    {0,1,3,7,4,5,2,6}, {0,1,7,6,4,5,3,2}, {0,1,6,2,4,5,7,3}  // B,B2,B'
};

// Device constant: edge permutation for each move (dest to src)
__device__ __constant__ int edgePermutation[18][12] = {
    {3,0,1,2,4,5,6,7,8,9,10,11}, {2,3,0,1,4,5,6,7,8,9,10,11}, {1,2,3,0,4,5,6,7,8,9,10,11}, // U,U2,U'
    {8,1,2,3,11,5,6,7,4,9,10,0}, {4,1,2,3,0,5,6,7,11,9,10,8}, {11,1,2,3,8,5,6,7,0,9,10,4}, // R,R2,R'
    {0,9,2,3,4,8,6,7,1,5,10,11}, {0,5,2,3,4,1,6,7,9,8,10,11}, {0,8,2,3,4,9,6,7,5,1,10,11}, // F,F2,F'
    {0,1,2,3,5,6,7,4,8,9,10,11}, {0,1,2,3,6,7,4,5,8,9,10,11}, {0,1,2,3,7,4,5,6,8,9,10,11}, // D,D2,D'
    {0,1,10,3,4,5,9,7,8,2,6,11}, {0,1,6,3,4,5,2,7,8,10,9,11}, {0,1,9,3,4,5,10,7,8,6,2,11}, // L,L2,L'
    {0,1,2,11,4,5,6,10,8,9,3,7}, {0,1,2,7,4,5,6,3,8,9,11,10}, {0,1,2,10,4,5,6,11,8,9,7,3} // B,B2,B'
};

// Device constant: target mask for middle-slice
__device__ __constant__ u16 targetMiddleSliceMask = 0b000001010101;

// Apply corner-permutation move (ignore orientation)
__device__ u32 applyCornerPermutationMoveGpu(u32 state, int moveIndex) {
    u32 result = 0;
    for (int dest = 0; dest < 8; ++dest) {
        int src = cornerPermutation[moveIndex][dest];
        u32 piece = (state >> (3 * src)) & 7u;
        result |= piece << (3 * dest);
    }
    return result;
}

// Apply an edge-slice move (1 bit per edge)
__device__ u16 applyEdgeSliceMoveGpu(u16 state, int moveIndex) {
    u16 result = 0;
    for (int dest = 0; dest < 12; ++dest) {
        int src = edgePermutation[moveIndex][dest];
        u16 bit = (state >> src) & 1u;
        result |= bit << dest;
    }
    return result;
}

// Check if corner permutation has even parity
__device__ bool isCornerParityEven(u32 cornerState) {
    int corners[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        corners[i] = (cornerState >> (3 * i)) & 7u;
    }
    int inversionCount = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            inversionCount ^= (corners[i] > corners[j]);
        }
    }
    return inversionCount == 0;
}

// Verify corner pairing groups
__device__ bool areCornerPairGroupsValid(u32 cornerState) {
    unsigned maskAccumulator = 0;
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        u32 piece = (cornerState >> (3 * slot)) & 7u;
        maskAccumulator |= ((piece ^ slot) & 5u);
    }
    return maskAccumulator == 0;
}

// Kernel: brute-force search for stage 3 solution
__global__ void bruteForceStage3Kernel(
    u32 startCornerState,
    u16 startMiddleSliceState,
    int searchDepth,
    int *d_solutionMoveBuffer,
    int *d_solutionFoundFlag
) {
    unsigned long long threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalSequences = 1ULL;

    for (int i = 0; i < searchDepth; ++i) totalSequences *= 10ULL;
    if (threadIndex >= totalSequences) return;

    u32 cornerState = startCornerState;
    u16 middleSliceState = startMiddleSliceState;

    int localMoveSequence[STAGE3_MAX_DEPTH];
    unsigned long long sequenceCode = threadIndex;
    int previousFaceIndex = -1;

    for (int step = 0; step < searchDepth; ++step) {
        int selectionIndex = sequenceCode % 10;
        sequenceCode /= 10;
        int moveIndex = deviceAllowedMoves[selectionIndex];
        int faceIndex = moveIndex / 3;
        if (faceIndex == previousFaceIndex) return;
        previousFaceIndex = faceIndex;
        localMoveSequence[step] = moveIndex;
        cornerState = applyCornerPermutationMoveGpu(cornerState, moveIndex);
        middleSliceState = applyEdgeSliceMoveGpu(middleSliceState, moveIndex);
    }

    bool middleSliceCorrect = (middleSliceState & targetMiddleSliceMask) == targetMiddleSliceMask;
    if (middleSliceCorrect
        && isCornerParityEven(cornerState)
        && areCornerPairGroupsValid(cornerState)) {
        if (atomicExch(d_solutionFoundFlag, 1) == 0) {
            for (int i = 0; i < searchDepth; ++i) {
                d_solutionMoveBuffer[i] = localMoveSequence[i];
            }
        }
    }
}

// Host: solve stage 3 using iterative deepening search on GPU
std::vector<std::string> solveStage3(u64 packedCornerState, u64 packedEdgeState) {
    u32 h_initialCornerState = 0;
    for (int i = 0; i < 8; ++i) {
        u32 piece = (packedCornerState >> (5 * i)) & 7u;
        h_initialCornerState |= piece << (3 * i);
    }

    u16 h_initialMiddleSliceState = 0;
    for (int i = 0; i < 12; ++i) {
        u32 piece = (packedEdgeState >> (5 * i)) & 31u;
        bool inMiddle = (piece == 0 || piece == 2 || piece == 4 || piece == 6);
        h_initialMiddleSliceState |= u16(inMiddle) << i;
    }

    int *d_solutionMoveBuffer;
    int *d_solutionFoundFlag;
    cudaMalloc(&d_solutionMoveBuffer, sizeof(int) * STAGE3_MAX_DEPTH);
    cudaMalloc(&d_solutionFoundFlag, sizeof(int));
    int h_initialFlag = 0;
    cudaMemcpy(d_solutionFoundFlag, &h_initialFlag, sizeof(h_initialFlag), cudaMemcpyHostToDevice);
    int h_solutionMoveIndices[STAGE3_MAX_DEPTH] = {0};
    int h_solutionDepthFound = 0;
    auto h_searchStartTime = std::chrono::steady_clock::now();

    for (int depth = 1; depth <= STAGE3_MAX_DEPTH; ++depth) {
        unsigned long long totalSequences = 1ULL;
        for (int i = 0; i < depth; ++i) totalSequences *= 10ULL;
        int threadsPerBlock = 256;
        int blockCount = (totalSequences + threadsPerBlock - 1) / threadsPerBlock;
        bruteForceStage3Kernel<<<blockCount, threadsPerBlock>>>(
            h_initialCornerState,
            h_initialMiddleSliceState,
            depth,
            d_solutionMoveBuffer,
            d_solutionFoundFlag
        );
        cudaDeviceSynchronize();
        int h_foundFlag = 0;
        cudaMemcpy(&h_foundFlag, d_solutionFoundFlag, sizeof(h_foundFlag), cudaMemcpyDeviceToHost);
        if (h_foundFlag) {
            h_solutionDepthFound = depth;
            cudaMemcpy(h_solutionMoveIndices, d_solutionMoveBuffer, sizeof(int) * depth, cudaMemcpyDeviceToHost);
            break;
        }
    }

    auto h_searchEndTime = std::chrono::steady_clock::now();
    auto h_elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(h_searchEndTime - h_searchStartTime).count();

    std::cout << "Stage 3 (Double Move Reduction) solved in " << h_elapsedMs << " ms\n";
    cudaFree(d_solutionMoveBuffer);
    cudaFree(d_solutionFoundFlag);

    static const char* moveNameList[18] = {
        "U","U2","U'","R","R2","R'",
        "F","F2","F'","D","D2","D'",
        "L","L2","L'","B","B2","B'"
    };

    std::vector<std::string> h_solutionMoves;
    h_solutionMoves.reserve(h_solutionDepthFound);
    for (int i = 0; i < h_solutionDepthFound; ++i) {
        h_solutionMoves.emplace_back(moveNameList[h_solutionMoveIndices[i]]);
    }
    return h_solutionMoves;
}
