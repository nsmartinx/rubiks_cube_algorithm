// stage2.cu
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using u64 = uint64_t;
using u16 = uint16_t;

#define MAX_DEPTH_STAGE_2 10

// Device constant: allowed Thistlethwaite G1 moves (preserves edge orientation)
__device__ __constant__ int d_allowedMoves[14] = {
    0, 1, 2,    // U, U2, U'
    3, 4, 5,    // R, R2, R'
    9, 10, 11,  // D, D2, D'
    12, 13, 14, // L, L2, L'
    7,          // F2
    16          // B2
};

// Device constant: corner permutation for each move (18 * 8 corners)
__device__ __constant__ int d_cornerPermutation[18][8] = {
    {3,0,1,2,4,5,6,7}, {2,3,0,1,4,5,6,7}, {1,2,3,0,4,5,6,7},
    {4,1,2,0,7,5,6,3}, {7,1,2,4,3,5,6,0}, {3,1,2,7,0,5,6,4},
    {1,5,2,3,0,4,6,7}, {5,4,2,3,1,0,6,7}, {4,0,2,3,5,1,6,7},
    {0,1,2,3,5,6,7,4}, {0,1,2,3,6,7,4,5}, {0,1,2,3,7,4,5,6},
    {0,2,6,3,4,1,5,7}, {0,6,5,3,4,2,1,7}, {0,5,1,3,4,6,2,7},
    {0,1,3,7,4,5,2,6}, {0,1,7,6,4,5,3,2}, {0,1,6,2,4,5,7,3}
};

// Device constant: corner twist for each move (18 * 8 corners)
__device__ __constant__ int d_cornerTwist[18][8] = {
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {2,0,0,1,1,0,0,2}, {0,0,0,0,0,0,0,0}, {2,0,0,1,1,0,0,2},
    {1,2,0,0,2,1,0,0}, {0,0,0,0,0,0,0,0}, {1,2,0,0,2,1,0,0},
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
    {0,1,2,0,0,2,1,0}, {0,0,0,0,0,0,0,0}, {0,1,2,0,0,2,1,0},
    {0,0,1,2,0,0,2,1}, {0,0,0,0,0,0,0,0}, {0,0,1,2,0,0,2,1}
};

// Device constant: edge permutation for each move (18 * 12 edges)
__device__ __constant__ int d_edgePermutation[18][12] = {
    {3,0,1,2,4,5,6,7,8,9,10,11}, {2,3,0,1,4,5,6,7,8,9,10,11}, {1,2,3,0,4,5,6,7,8,9,10,11},
    {8,1,2,3,11,5,6,7,4,9,10,0}, {4,1,2,3,0,5,6,7,11,9,10,8}, {11,1,2,3,8,5,6,7,0,9,10,4},
    {0,9,2,3,4,8,6,7,1,5,10,11}, {0,5,2,3,4,1,6,7,9,8,10,11}, {0,8,2,3,4,9,6,7,5,1,10,11},
    {0,1,2,3,5,6,7,4,8,9,10,11}, {0,1,2,3,6,7,4,5,8,9,10,11}, {0,1,2,3,7,4,5,6,8,9,10,11},
    {0,1,10,3,4,5,9,7,8,2,6,11}, {0,1,6,3,4,5,2,7,8,10,9,11}, {0,1,9,3,4,5,10,7,8,6,2,11},
    {0,1,2,11,4,5,6,10,8,9,3,7}, {0,1,2,7,4,5,6,3,8,9,11,10}, {0,1,2,10,4,5,6,11,8,9,7,3}
};

// Device constant: The 4 edges that must be in the E slice
__device__ __constant__ u16 d_targetSliceMask = 0b1111'0000'0000;

// Host notation for moves
static const char* h_moveNotation[18] = {
    "U","U2","U'",
    "R","R2","R'",
    "F","F2","F'",
    "D","D2","D'",
    "L","L2","L'",
    "B","B2","B'"
};

// Device: apply corner-twist move
__device__ u16 applyCornerTwistMoveGpu(u16 currentCornerState, int moveIndex) {
    u16 newCornerState = 0;
    for (int cornerDestination = 0; cornerDestination < 8; ++cornerDestination) {
        int cornerSource = d_cornerPermutation[moveIndex][cornerDestination];
        int orientation = (currentCornerState >> (2 * cornerSource)) & 0x3;
        int twist = d_cornerTwist[moveIndex][cornerDestination];
        int newOrientation = (orientation + twist) % 3;
        newCornerState |= u16(newOrientation) << (2 * cornerDestination);
    }
    return newCornerState;
}

// Device: apply slice-membership move
__device__ u16 applySliceMembershipMoveGpu(u16 currentSliceMask, int moveIndex) {
    u16 newSliceMask = 0;
    for (int edgeDestination = 0; edgeDestination < 12; ++edgeDestination) {
        int edgeSource = d_edgePermutation[moveIndex][edgeDestination];
        u16 bit = (currentSliceMask >> edgeSource) & 0x1;
        newSliceMask |= bit << edgeDestination;
    }
    return newSliceMask;
}

// Kernel: brute-force stage 2 sequences up to given depth
__global__ void bruteForceStage2Kernel(
    u16 startCornerState,
    u16 startSliceMask,
    int searchDepth,
    int* d_solutionMoveIndicesBuffer,
    int* d_solutionFoundFlag
) {
    unsigned long long threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalSequences = 1;
    for (int depthIndex = 0; depthIndex < searchDepth; ++depthIndex) {
        totalSequences *= 14ULL;
    }
    if (threadId >= totalSequences) {
        return;
    }

    u16 cornerState = startCornerState;
    u16 sliceMask = startSliceMask;
    int localMoveSequence[MAX_DEPTH_STAGE_2];
    unsigned long long code = threadId;
    int previousFace = -1;

    for (int step = 0; step < searchDepth; ++step) {
        int moveSelection = code % 14;
        code /= 14;
        int moveIndex = d_allowedMoves[moveSelection];
        int faceIndex = moveIndex / 3;
        if (faceIndex == previousFace) {
            return;
        }
        previousFace = faceIndex;
        localMoveSequence[step] = moveIndex;
        cornerState = applyCornerTwistMoveGpu(cornerState, moveIndex);
        sliceMask = applySliceMembershipMoveGpu(sliceMask, moveIndex);
    }

    if (cornerState == 0 && (sliceMask & d_targetSliceMask) == d_targetSliceMask) {
        if (atomicExch(d_solutionFoundFlag, 1) == 0) {
            for (int index = 0; index < searchDepth; ++index) {
                d_solutionMoveIndicesBuffer[index] = localMoveSequence[index];
            }
        }
    }
}

// Host: solve stage 2 by brute force on GPU
std::vector<std::string> solveStage2(u64 cornerStateValue, u64 edgeStateValue) {
    // Pack corner orientation bits
    u16 startCornerState = 0;
    for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
        int packedCorner = (cornerStateValue >> (5 * cornerIndex)) & 0x1F;
        int cornerOrientation = (packedCorner >> 3) & 0x3;
        startCornerState |= u16(cornerOrientation) << (2 * cornerIndex);
    }

    // Pack slice-membership bits
    u16 startSliceMask = 0;
    for (int edgeIndex = 0; edgeIndex < 12; ++edgeIndex) {
        int packedEdge = (edgeStateValue >> (5 * edgeIndex)) & 0x1F;
        int edgePiece = packedEdge & 0xF;
        bool isInSlice = (edgePiece >= 8 && edgePiece <= 11);
        startSliceMask |= u16(isInSlice) << edgeIndex;
    }

    // Allocate GPU buffers
    int* d_solutionMoveIndicesBuffer; int* d_solutionFoundFlag;
    cudaMalloc(&d_solutionMoveIndicesBuffer, sizeof(int) * MAX_DEPTH_STAGE_2);
    cudaMalloc(&d_solutionFoundFlag, sizeof(int));
    int hostZeroFlag = 0; cudaMemcpy(d_solutionFoundFlag, &hostZeroFlag, sizeof(hostZeroFlag), cudaMemcpyHostToDevice);

    // Iterative deepening search
    int hostSolutionMoveIndices[MAX_DEPTH_STAGE_2] = {0};
    int foundSolutionDepth = 0;
    auto startTimestamp = std::chrono::steady_clock::now();

    for (int searchDepth = 1; searchDepth <= MAX_DEPTH_STAGE_2; ++searchDepth) {
        unsigned long long totalSequences = 1;
        for (int index = 0; index < searchDepth; ++index) {
            totalSequences *= 14ULL;
        }
        int threadsPerBlockCount = 256;
        int deviceBlockCount = int((totalSequences + threadsPerBlockCount - 1) / threadsPerBlockCount);

        bruteForceStage2Kernel<<<deviceBlockCount, threadsPerBlockCount>>>(
            startCornerState,
            startSliceMask,
            searchDepth,
            d_solutionMoveIndicesBuffer,
            d_solutionFoundFlag
        );
        cudaDeviceSynchronize();

        int hostFoundFlag = 0; cudaMemcpy(&hostFoundFlag, d_solutionFoundFlag, sizeof(hostFoundFlag), cudaMemcpyDeviceToHost);
        if (hostFoundFlag) {
            foundSolutionDepth = searchDepth;
            cudaMemcpy(hostSolutionMoveIndices, d_solutionMoveIndicesBuffer, sizeof(int) * searchDepth, cudaMemcpyDeviceToHost);
            break;
        }
    }
    auto endTimestamp = std::chrono::steady_clock::now();
    auto elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTimestamp - startTimestamp).count();
    std::cout << "Stage 2 (Domino Reduction) solved in " << elapsedMilliseconds << " ms\n";

    cudaFree(d_solutionMoveIndicesBuffer); cudaFree(d_solutionFoundFlag);

    // Build result strings
    std::vector<std::string> solutionMoves;
    solutionMoves.reserve(foundSolutionDepth);
    for (int index = 0; index < foundSolutionDepth; ++index) {
        solutionMoves.emplace_back(h_moveNotation[hostSolutionMoveIndices[index]]);
    }
    return solutionMoves;
}
