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
    {0,1,2,11,4,5,6,10,8,9,3,7}, {0,1,2,7,4,5,6,3,8,9,11,10}, {0,1,2,10,4,5,6,11,8,9,7,3}  // B,B2,B'
};

// Device constant: target mask for middle-slice
__device__ __constant__ u16 d_targetMiddleSliceMask = 0b0000'0101'0101;

// Apply corner-permutation move (ignore orientation)
__device__ u32 applyCornerPermutationMoveGpu(u32 cornerState, int moveIndex) {
    u32 permutedCornerState = 0;
    for (int destinationSlot = 0; destinationSlot < 8; ++destinationSlot) {
        int sourceSlot = cornerPermutation[moveIndex][destinationSlot];
        u32 cornerPiece = (cornerState >> (3 * sourceSlot)) & 7u;
        permutedCornerState |= cornerPiece << (3 * destinationSlot);
    }
    return permutedCornerState;
}

// Apply an edge-slice move (1 bit per edge)
__device__ u16 applyEdgeMoveGpu(u16 edgeState, int moveIndex) {
    u16 permutedEdgeState = 0;
    for (int destinationSlot = 0; destinationSlot < 12; ++destinationSlot) {
        int sourceSlot = edgePermutation[moveIndex][destinationSlot];
        u16 edgeBit = (edgeState >> sourceSlot) & 1u;
        permutedEdgeState |= edgeBit << destinationSlot;
    }
    return permutedEdgeState;
}

// Check if corner permutation has even parity
__device__ bool isCornerParityEven(u32 cornerState) {
    int cornerList[8];
    #pragma unroll
    for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
        cornerList[cornerIndex] = (cornerState >> (3 * cornerIndex)) & 7u;
    }
    int inversionParity = 0;
    #pragma unroll
    for (int outerIndex = 0; outerIndex < 8; ++outerIndex) {
        for (int innerIndex = outerIndex + 1; innerIndex < 8; ++innerIndex) {
            inversionParity ^= (cornerList[outerIndex] > cornerList[innerIndex]);
        }
    }
    return inversionParity == 0;
}

// Verify corner pairing groups
__device__ bool areCornerPairGroupsValid(u32 cornerState) {
    unsigned maskAccumulator = 0;
    #pragma unroll
    for (int cornerSlot = 0; cornerSlot < 8; ++cornerSlot) {
        u32 cornerPiece = (cornerState >> (3 * cornerSlot)) & 7u;
        maskAccumulator |= ((cornerPiece ^ cornerSlot) & 5u);
    }
    return maskAccumulator == 0;
}

// Kernel: brute-force search for stage 3 solution
__global__ void bruteForceStage3Kernel(
    u32 d_startCornerState,
    u16 d_startMiddleSliceState,
    int d_searchDepth,
    int *d_solutionMoveBuffer,
    int *d_solutionFoundFlag
) {
    unsigned long long threadGlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long totalSequenceCount = 1ULL;
    for (int depthIteration = 0; depthIteration < d_searchDepth; ++depthIteration) {
        totalSequenceCount *= 10ULL;
    }
    if (threadGlobalIndex >= totalSequenceCount) return;

    u32 localCornerState = d_startCornerState;
    u16 localMiddleSliceState = d_startMiddleSliceState;

    int localMoveSequence[STAGE3_MAX_DEPTH];
    unsigned long long encodedSequenceNumber = threadGlobalIndex;
    int previousFaceIdentifier = -1;

    for (int stepIndex = 0; stepIndex < d_searchDepth; ++stepIndex) {
        int selectionIndex = encodedSequenceNumber % 10;
        encodedSequenceNumber /= 10;
        int selectedMoveIndex = deviceAllowedMoves[selectionIndex];
        int faceIdentifier = selectedMoveIndex / 3;
        if (faceIdentifier == previousFaceIdentifier) return;
        previousFaceIdentifier = faceIdentifier;
        localMoveSequence[stepIndex] = selectedMoveIndex;
        localCornerState = applyCornerPermutationMoveGpu(localCornerState, selectedMoveIndex);
        localMiddleSliceState = applyEdgeMoveGpu(localMiddleSliceState, selectedMoveIndex);
    }

    bool isMiddleSliceCorrect = (localMiddleSliceState & d_targetMiddleSliceMask) == d_targetMiddleSliceMask;

    if (isMiddleSliceCorrect && isCornerParityEven(localCornerState) && areCornerPairGroupsValid(localCornerState)) {
        if (atomicExch(d_solutionFoundFlag, 1) == 0) {
            for (int depthIteration = 0; depthIteration < d_searchDepth; ++depthIteration) {
                d_solutionMoveBuffer[depthIteration] = localMoveSequence[depthIteration];
            }
        }
    }
}

// Host: solve stage 3 using iterative deepening search on GPU
std::vector<std::string> solveStage3(u64 h_packedCornerState, u64 h_packedEdgeState) {
    u32 h_initialCornerState = 0;
    for (int cornerSlotIndex = 0; cornerSlotIndex < 8; ++cornerSlotIndex) {
        u32 cornerPiece = (h_packedCornerState >> (5 * cornerSlotIndex)) & 7u;
        h_initialCornerState |= cornerPiece << (3 * cornerSlotIndex);
    }

    u16 h_initialMiddleSliceState = 0;
    for (int edgeSlotIndex = 0; edgeSlotIndex < 12; ++edgeSlotIndex) {
        u32 edgePiece = (h_packedEdgeState >> (5 * edgeSlotIndex)) & 31u;
        bool isInMiddleSlice = (edgePiece == 0 || edgePiece == 2 || edgePiece == 4 || edgePiece == 6);
        h_initialMiddleSliceState |= u16(isInMiddleSlice) << edgeSlotIndex;
    }

    int *d_solutionMoveBuffer = nullptr;
    int *d_solutionFoundFlag = nullptr;
    cudaMalloc(&d_solutionMoveBuffer, sizeof(int) * STAGE3_MAX_DEPTH);
    cudaMalloc(&d_solutionFoundFlag, sizeof(int));

    int h_initialFlag = 0;
    cudaMemcpy(d_solutionFoundFlag, &h_initialFlag, sizeof(h_initialFlag), cudaMemcpyHostToDevice);

    int h_solutionMoveIndices[STAGE3_MAX_DEPTH] = {0};
    int h_solutionDepthFound = 0;
    auto h_searchStartTime = std::chrono::steady_clock::now();

    for (int depth = 1; depth <= STAGE3_MAX_DEPTH; ++depth) {
        unsigned long long totalSequences = 1ULL;
        for (int depthIteration = 0; depthIteration < depth; ++depthIteration) {
            totalSequences *= 10ULL;
        }
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
    for (int moveIndex = 0; moveIndex < h_solutionDepthFound; ++moveIndex) {
        h_solutionMoves.emplace_back(moveNameList[h_solutionMoveIndices[moveIndex]]);
    }
    return h_solutionMoves;
}
