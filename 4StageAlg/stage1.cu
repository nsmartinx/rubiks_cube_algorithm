// stage1.cu
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using u64 = uint64_t;
using u16 = uint16_t;

#define MAX_DEPTH_STAGE_1 7

// Device constant: edge permutation for each move (18 moves × 12 edges)
__device__ __constant__ int d_edgePermutation[18][12] = {
    // U
    {3,0,1,2,4,5,6,7,8,9,10,11},
    // U2
    {2,3,0,1,4,5,6,7,8,9,10,11},
    // U'
    {1,2,3,0,4,5,6,7,8,9,10,11},

    // R
    {8,1,2,3,11,5,6,7,4,9,10,0},
    // R2
    {4,1,2,3,0,5,6,7,11,9,10,8},
    // R'
    {11,1,2,3,8,5,6,7,0,9,10,4},

    // F
    {0,9,2,3,4,8,6,7,1,5,10,11},
    // F2
    {0,5,2,3,4,1,6,7,9,8,10,11},
    // F'
    {0,8,2,3,4,9,6,7,5,1,10,11},

    // D
    {0,1,2,3,5,6,7,4,8,9,10,11},
    // D2
    {0,1,2,3,6,7,4,5,8,9,10,11},
    // D'
    {0,1,2,3,7,4,5,6,8,9,10,11},

    // L
    {0,1,10,3,4,5,9,7,8,2,6,11},
    // L2
    {0,1,6,3,4,5,2,7,8,10,9,11},
    // L'
    {0,1,9,3,4,5,10,7,8,6,2,11},

    // B
    {0,1,2,11,4,5,6,10,8,9,3,7},
    // B2
    {0,1,2,7,4,5,6,3,8,9,11,10},
    // B'
    {0,1,2,10,4,5,6,11,8,9,7,3}
};

// Device constant: edge flip mask for each move (18 moves)
__device__ __constant__ u16 d_edgeFlipMask[18] = {
    // U
    0b0000'0000'0000,
    // U2
    0b0000'0000'0000,
    // U'
    0b0000'0000'0000,

    // R
    0b0000'0000'0000,
    // R2
    0b0000'0000'0000,
    // R'
    0b0000'0000'0000,

    // F
    0b0011'0010'0010,
    // F2
    0b0000'0000'0000,
    // F'
    0b0011'0010'0010,

    // D
    0b0000'0000'0000,
    // D2
    0b0000'0000'0000,
    // D'
    0b0000'0000'0000,

    // L
    0b0000'0000'0000,
    // L2
    0b0000'0000'0000,
    // L'
    0b0000'0000'0000,

    // B
    0b1100'1000'1000,
    // B2
    0b0000'0000'0000,
    // B'
    0b1100'1000'1000
};

// Host notation for moves
static const char* moveNotationNames[18] = {
    "U","U2","U'",
    "R","R2","R'",
    "F","F2","F'",
    "D","D2","D'",
    "L","L2","L'",
    "B","B2","B'"
};

// Device: apply a single move to edge orientation state
__device__ u16 applyMoveGpu(u16 currentState, int moveIndex) {
    u16 newState = 0;
    u16 flipMask = d_edgeFlipMask[moveIndex];
    for (int destination = 0; destination < 12; ++destination) {
        int source = d_edgePermutation[moveIndex][destination];
        int orientation = (currentState >> source) & 1;
        if ((flipMask >> source) & 1) {
            orientation ^= 1;
        }
        newState |= (u16(orientation) << destination);
    }
    return newState;
}

// Kernel: brute-force stage 1 (edge orientation) sequences up to given depth
__global__ void bruteForceStage1Kernel(
    u16 startState,
    int depth,
    int* d_solutionBuffer,
    int* d_solutionFoundFlag
) {
    unsigned long long threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long totalSequences = 1;
    for (int i = 0; i < depth; ++i) {
        totalSequences *= 18ULL;
    }
    if (threadId >= totalSequences) {
        return;
    }

    u16 currentState = startState;
    int moveSequence[MAX_DEPTH_STAGE_1];
    int previousFace = -1;
    unsigned long long code = threadId;

    // Decode sequence and apply moves
    for (int depthIndex = 0; depthIndex < depth; ++depthIndex) {
        int move = code % 18;
        code /= 18;
        int face = move / 3;
        if (face == previousFace) {
            return;
        }
        previousFace = face;
        moveSequence[depthIndex] = move;
        currentState = applyMoveGpu(currentState, move);
    }

    // Check if solved
    if (currentState == 0) {
        if (atomicExch(d_solutionFoundFlag, 1) == 0) {
            for (int i = 0; i < depth; ++i) {
                d_solutionBuffer[i] = moveSequence[i];
            }
        }
    }
}

// Host: solve stage 1 by brute force on GPU
std::vector<std::string> solveStage1(u64 edgeState) {
    // Extract edge orientation bits to start state
    u16 startEdgeOrientation = 0;
    for (int index = 0; index < 12; ++index) {
        int packedPiece = (edgeState >> (5 * index)) & 0x1F;
        int orientation = (packedPiece >> 4) & 1;
        startEdgeOrientation |= u16(orientation << index);
    }
    if (startEdgeOrientation == 0) {
        return {};
    }

    // Allocate GPU buffers
    int* d_solutionBuffer;
    int* d_solutionFoundFlag;
    cudaMalloc(&d_solutionBuffer, sizeof(int) * MAX_DEPTH_STAGE_1);
    cudaMalloc(&d_solutionFoundFlag, sizeof(int));
    int zeroFlag = 0;
    cudaMemcpy(d_solutionFoundFlag, &zeroFlag, sizeof(zeroFlag), cudaMemcpyHostToDevice);

    // Brute-force on increasing depth
    int h_solutionIndices[MAX_DEPTH_STAGE_1] = {0};
    int foundSolutionDepth = 0;
    auto startTimer = std::chrono::steady_clock::now();

    for (int depth = 1; depth <= MAX_DEPTH_STAGE_1; ++depth) {
        unsigned long long totalSequences = 1;
        for (int i = 0; i < depth; ++i) {
            totalSequences *= 18ULL;
        }
        int threadsPerBlockCount = 256;
        int blockCount = (totalSequences + threadsPerBlockCount - 1) / threadsPerBlockCount;

        bruteForceStage1Kernel<<<blockCount, threadsPerBlockCount>>>(
            startEdgeOrientation,
            depth,
            d_solutionBuffer,
            d_solutionFoundFlag
        );
        cudaDeviceSynchronize();

        int foundFlag = 0;
        cudaMemcpy(&foundFlag, d_solutionFoundFlag, sizeof(foundFlag), cudaMemcpyDeviceToHost);
        if (foundFlag) {
            foundSolutionDepth = depth;
            cudaMemcpy(h_solutionIndices, d_solutionBuffer, sizeof(int) * depth, cudaMemcpyDeviceToHost);
            break;
        }
    }
    auto endTimer = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer - startTimer).count();
    std::cout << "Stage 1 (edge orientation) solved in " << elapsedMs << " ms\n";

    cudaFree(d_solutionBuffer);
    cudaFree(d_solutionFoundFlag);

    // Build result strings
    std::vector<std::string> solutionMoves;
    solutionMoves.reserve(foundSolutionDepth);
    for (int i = 0; i < foundSolutionDepth; ++i) {
        solutionMoves.emplace_back(
            moveNotationNames[h_solutionIndices[i]]
        );
    }
    return solutionMoves;
}
