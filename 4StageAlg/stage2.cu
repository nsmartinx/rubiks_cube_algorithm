#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using u64     = uint64_t;
using COState = uint16_t;
using ESState = uint16_t;

#define MAX_SEARCH_DEPTH 10

// —–– allowed Thistlethwaite G1 moves (preserves edge‐orientation) ——
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
  
// —–– edge perm table for updating “slice‐membership” ——
__device__ __constant__ int ePerm[18][12] = {
  {3,0,1,2,4,5,6,7,8,9,10,11}, {2,3,0,1,4,5,6,7,8,9,10,11}, {1,2,3,0,4,5,6,7,8,9,10,11},
  {8,1,2,3,11,5,6,7,4,9,10,0}, {4,1,2,3,0,5,6,7,11,9,10,8}, {11,1,2,3,8,5,6,7,0,9,10,4},
  {0,9,2,3,4,8,6,7,1,5,10,11}, {0,5,2,3,4,1,6,7,9,8,10,11}, {0,8,2,3,4,9,6,7,5,1,10,11},
  {0,1,2,3,5,6,7,4,8,9,10,11}, {0,1,2,3,6,7,4,5,8,9,10,11}, {0,1,2,3,7,4,5,6,8,9,10,11},
  {0,1,10,3,4,5,9,7,8,2,6,11},{0,1,6,3,4,5,2,7,8,10,9,11},{0,1,9,3,4,5,10,7,8,6,2,11},
  {0,1,2,11,4,5,6,10,8,9,3,7},{0,1,2,7,4,5,6,3,8,9,11,10},{0,1,2,10,4,5,6,11,8,9,7,3}
};

// target mask: bits 8–11 must all be 1
__device__ __constant__ ESState targetSliceMask = (1<<8)|(1<<9)|(1<<10)|(1<<11);

// apply a corner‐twist move
__device__ COState applyCOMoveGPU(COState s, int m) {
    COState t = 0;
    // exactly the same structure as the CPU code:
    for (int dst = 0; dst < 8; ++dst) {
        int src = coPerm[m][dst];                            // dst→src
        int ori = (s >> (2*src)) & 0x3;                      // read old orientation
        int nori = (ori + coTwist[m][dst]) % 3;              // twist for that dst
        t |= COState(nori) << (2*dst);                       // write new orientation
    }
    return t;
}


// apply a slice‐membership move
__device__ ESState applyESMoveGPU(ESState s, int m) {
    ESState t = 0;
    for (int dst = 0; dst < 12; ++dst) {
        int src = ePerm[m][dst];
        ESState bit = (s >> src) & 0x1;
        t |= bit << dst;
    }
    return t;
}

// brute‑force kernel now tests BOTH CORNER‐ORIENTATION == 0
// and all 4 slice‐edges in slots 8..11
__global__ void bruteForceCOKernel(
    COState    startCoro,
    ESState    startMask,
    int        depth,
    int       *solutionBuffer,
    int       *solutionFoundFlag
) {
    unsigned long long tid = blockIdx.x*blockDim.x + threadIdx.x;
    // total = 14^depth
    unsigned long long total = 1;
    for (int i = 0; i < depth; ++i) total *= 14ULL;
    if (tid >= total) return;

    COState coro = startCoro;
    ESState mask = startMask;
    int seq[MAX_SEARCH_DEPTH];
    unsigned long long code = tid;
    int prevFace = -1;

    for (int d = 0; d < depth; ++d) {
        int sel  = code % 14; 
        code    /= 14;
        int m    = allowedMoves[sel];
        int face = m/3;
        if (face == prevFace) return;
        prevFace = face;
        seq[d] = m;
        // update both state components
        coro = applyCOMoveGPU(coro, m);
        mask = applyESMoveGPU(mask, m);
    }

    // success if corners all zero AND slice‐mask bits 8..11 set
    if (coro == 0 && (mask & targetSliceMask) == targetSliceMask) {
        if (atomicExch(solutionFoundFlag, 1) == 0) {
            for (int i = 0; i < depth; ++i)
                solutionBuffer[i] = seq[i];
        }
    }
}

std::vector<std::string> solveStage2(u64 cornerState, u64 edgeState) {
    // 1) pack corner‐orientation
    COState coro = 0;
    for (int i = 0; i < 8; ++i) {
        int packed = (cornerState >> (5*i)) & 0x1F;
        int ori    = (packed >> 3) & 0x3;
        coro |= COState(ori) << (2*i);
    }
    if (coro == 0) {
        // but if edges not yet in slice, still need to search
    }

    // 2) pack edge “slice‐membership”:
    //    bit i = 1 if the piece in slot i is one of {8,9,10,11}
    ESState mask = 0;
    for (int i = 0; i < 12; ++i) {
        int packed = (edgeState >> (5*i)) & 0x1F;
        int piece  = packed & 0xF;
        bool isSlice = (piece >= 8 && piece <= 11);
        mask |= ESState(isSlice) << i;
    }

    // 3) GPU buffers
    int *d_sol, *d_flag;
    cudaMalloc(&d_sol,  sizeof(int)*MAX_SEARCH_DEPTH);
    cudaMalloc(&d_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_flag, &zero, sizeof(zero), cudaMemcpyHostToDevice);

    // 4) iterative deepening
    auto t0 = std::chrono::steady_clock::now();
    int hostSol[MAX_SEARCH_DEPTH];
    int foundDepth = 0;
    for (int depth = 1; depth <= MAX_SEARCH_DEPTH; ++depth) {
        unsigned long long total = 1;
        for (int i = 0; i < depth; ++i) total *= 14ULL;
        int threadsPerBlock = 256;
        int blocks = int((total + threadsPerBlock - 1) / threadsPerBlock);

        bruteForceCOKernel<<<blocks,threadsPerBlock>>>(
            coro, mask, depth, d_sol, d_flag
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
    std::cout << "Stage 2 (CO + E‑slice) solved in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    cudaFree(d_sol);
    cudaFree(d_flag);

    // 5) translate to move notation
    const char* names[18] = {
        "U","U2","U'", "R","R2","R'", "F","F2","F'", 
        "D","D2","D'", "L","L2","L'", "B","B2","B'"
    };
    std::vector<std::string> result;
    result.reserve(foundDepth);
    for (int i = 0; i < foundDepth; ++i)
        result.emplace_back(names[hostSol[i]]);
    return result;
}
