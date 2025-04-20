#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using u64     = uint64_t;
using COState = uint16_t;

#define MAX_SEARCH_DEPTH 10
// Only the 12 allowed moves:
__device__ __constant__ int allowedMoves[14] = {
    0,1,2,    // U U2 U'
    3,4,5,    // R R2 R'
    9,10,11,  // D D2 D'
    12,13,14, // L L2 L'
    7,        // F2  
    16        // B2
};

// src→dst corner permutation:
__device__ __constant__ int coPerm[18][8] = {
  {3,0,1,2,4,5,6,7},{2,3,0,1,4,5,6,7},{1,2,3,0,4,5,6,7},
  {4,1,2,0,7,5,6,3},{7,1,2,4,3,5,6,0},{3,1,2,7,0,5,6,4},
  {1,5,2,3,0,4,6,7},{5,4,2,3,1,0,6,7},{4,0,2,3,5,1,6,7},
  {0,1,2,3,5,6,7,4},{0,1,2,3,6,7,4,5},{0,1,2,3,7,4,5,6},
  {0,2,6,3,4,1,5,7},{0,6,5,3,4,2,1,7},{0,5,1,3,4,6,2,7},
  {0,1,3,7,4,5,2,6},{0,1,7,6,4,5,3,2},{0,1,6,2,4,5,7,3}
};

// how much each move twists each source corner
__device__ __constant__ int coTwist[18][8] = {
  {0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},
  {2,0,0,1,1,0,0,2},{0,0,0,0,0,0,0,0},{2,0,0,1,1,0,0,2},
  {1,2,0,0,2,1,0,0},{0,0,0,0,0,0,0,0},{1,2,0,0,2,1,0,0},
  {0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},
  {0,1,2,0,0,2,1,0},{0,0,0,0,0,0,0,0},{0,1,2,0,0,2,1,0},
  {0,0,1,2,0,0,2,1},{0,0,0,0,0,0,0,0},{0,0,1,2,0,0,2,1},
};

// Apply a corner‐twist move to a packed COState
__device__ COState applyCOMoveGPU(COState s, int m) {
    COState t = 0;
    for (int src = 0; src < 8; ++src) {
        int ori = (s >> (2*src)) & 0x3;
        int dst = coPerm[m][src];
        ori = (ori + coTwist[m][dst]) % 3;
        t |= COState(ori) << (2*dst);
    }
    return t;
}

// brute‑force kernel
__global__ void bruteForceCOKernel(
    COState startState,
    int depth,
    int *solutionBuffer,
    int *solutionFoundFlag
) {
    unsigned long long tid =
      blockIdx.x * blockDim.x + threadIdx.x;
    // total sequences = 12^depth
    unsigned long long total = 1;
    for (int i = 0; i < depth; ++i) total *= 14ULL;
    if (tid >= total) return;

    COState state = startState;
    int seq[MAX_SEARCH_DEPTH];
    unsigned long long code = tid;
    int prevFace = -1;

    for (int d = 0; d < depth; ++d) {
        int sel = code % 14;
        code   /= 14;
        int m   = allowedMoves[sel];
        int face = m/3;
        if (face == prevFace) return;
        prevFace = face;
        seq[d] = m;
        state = applyCOMoveGPU(state, m);
    }

    // solved if all orientation bits zero
    if (state == 0) {
        if (atomicExch(solutionFoundFlag, 1) == 0) {
            for (int i = 0; i < depth; ++i)
                solutionBuffer[i] = seq[i];
        }
    }
}

std::vector<std::string> solveStage2(u64 cornerState) {
    // 1) Pack the corner‐orientation bits into a 16‑bit COState:
    //    cornerState has 5 bits per corner in slots 0..7:
    //      bits [5*i .. 5*i+2] = piece index (ignored here)
    //      bits [5*i+3 .. 5*i+4] = orientation (0,1,2)
    using COState = uint16_t;
    COState coro = 0;
    for (int i = 0; i < 8; ++i) {
        // extract the 5‑bit chunk for corner slot i
        int packed = (cornerState >> (5 * i)) & 0x1F;
        // orientation = bits 3..4
        int ori    = (packed >> 3) & 0x3;
        // store into bits [2*i .. 2*i+1] of coro
        coro |= COState(ori << (2 * i));
    }

    // 2) If already all zeros, we're done
    if (coro == 0) return {};

    // 3) Allocate GPU buffers
    int *d_sol, *d_flag;
    cudaMalloc(&d_sol,  sizeof(int)*MAX_SEARCH_DEPTH);
    cudaMalloc(&d_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_flag, &zero, sizeof(zero), cudaMemcpyHostToDevice);

    // 4) Iterative deepening brute‐force on the 12 allowed moves
    auto t0 = std::chrono::steady_clock::now();
    int hostSol[MAX_SEARCH_DEPTH];
    int foundDepth = 0;
    for (int depth = 1; depth <= MAX_SEARCH_DEPTH; ++depth) {
        // total = 12^depth
        unsigned long long total = 1;
        for (int i = 0; i < depth; ++i) total *= 12ULL;

        int threadsPerBlock = 256;
        int blocks = int((total + threadsPerBlock - 1) / threadsPerBlock);

        bruteForceCOKernel<<<blocks,threadsPerBlock>>>(
            coro, depth, d_sol, d_flag
        );
        cudaDeviceSynchronize();

        int found = 0;
        cudaMemcpy(&found, d_flag, sizeof(found), cudaMemcpyDeviceToHost);
        if (found) {
            foundDepth = depth;
            cudaMemcpy(hostSol, d_sol, sizeof(int)*depth, cudaMemcpyDeviceToHost);
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Stage 2 (CO) solved in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    cudaFree(d_sol);
    cudaFree(d_flag);

    // 5) Translate solution indices to notation
    const char* names[18] = {
        "U","U2","U'","R","R2","R'","F","F2","F'",
        "D","D2","D'","L","L2","L'","B","B2","B'"
    };

    std::vector<std::string> result;
    result.reserve(foundDepth);
    for (int i = 0; i < foundDepth; ++i) {
        result.emplace_back(names[hostSol[i]]);
    }
    return result;
}
