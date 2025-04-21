#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using u64     = uint64_t;
// pack 8 corners ×3 bits = 24 bits
using CPState = uint32_t;
// pack 12 edges ×1 bit  = 12 bits
using ESState = uint16_t;

#define MAX_STAGE3_DEPTH 13

// —— G₂ moves: preserve EO, CO, E‑slice —— 
__device__ __constant__ int allowedMoves[14] = {
    0,1,2,    // U, U2, U'
    3,4,5,    // R, R2, R'
    9,10,11,  // D, D2, D'
    12,13,14, // L, L2, L'
    7,        // F2
    16        // B2
};

// dst→src corner perm  
__device__ __constant__ int coPerm[18][8] = {
  {3,0,1,2,4,5,6,7},{2,3,0,1,4,5,6,7},{1,2,3,0,4,5,6,7},
  {4,1,2,0,7,5,6,3},{7,1,2,4,3,5,6,0},{3,1,2,7,0,5,6,4},
  {1,5,2,3,0,4,6,7},{5,4,2,3,1,0,6,7},{4,0,2,3,5,1,6,7},
  {0,1,2,3,5,6,7,4},{0,1,2,3,6,7,4,5},{0,1,2,3,7,4,5,6},
  {0,2,6,3,4,1,5,7},{0,6,5,3,4,2,1,7},{0,5,1,3,4,6,2,7},
  {0,1,3,7,4,5,2,6},{0,1,7,6,4,5,3,2},{0,1,6,2,4,5,7,3}
};

// dst→src edge perm  
__device__ __constant__ int ePerm[18][12] = {
  {3,0,1,2,4,5,6,7,8,9,10,11},{2,3,0,1,4,5,6,7,8,9,10,11},{1,2,3,0,4,5,6,7,8,9,10,11},
  {8,1,2,3,11,5,6,7,4,9,10,0},{4,1,2,3,0,5,6,7,11,9,10,8},{11,1,2,3,8,5,6,7,0,9,10,4},
  {0,9,2,3,4,8,6,7,1,5,10,11},{0,5,2,3,4,1,6,7,9,8,10,11},{0,8,2,3,4,9,6,7,5,1,10,11},
  {0,1,2,3,5,6,7,4,8,9,10,11},{0,1,2,3,6,7,4,5,8,9,10,11},{0,1,2,3,7,4,5,6,8,9,10,11},
  {0,1,10,3,4,5,9,7,8,2,6,11},{0,1,6,3,4,5,2,7,8,10,9,11},{0,1,9,3,4,5,10,7,8,6,2,11},
  {0,1,2,11,4,5,6,10,8,9,3,7},{0,1,2,7,4,5,6,3,8,9,11,10},{0,1,2,10,4,5,6,11,8,9,7,3}
};

// bit‑masks for checking slices
__device__ __constant__ ESState targetMSliceMask = (1<<0)|(1<<2)|(1<<4)|(1<<6);
__device__ __constant__ ESState targetSSliceMask = (1<<1)|(1<<3)|(1<<5)|(1<<7);

// apply corner‑perm (ignore orientation)
__device__ CPState applyCPMoveGPU(CPState s, int m) {
    CPState t = 0;
    for (int dst = 0; dst < 8; ++dst) {
        int src = coPerm[m][dst];
        CPState c = (s >> (3*src)) & 0x7;
        t |= c << (3*dst);
    }
    return t;
}

// apply a 1‑bit slice‑mask move
__device__ ESState applySliceMoveGPU(ESState s, int m) {
    ESState t = 0;
    for (int dst = 0; dst < 12; ++dst) {
        int src    = ePerm[m][dst];
        ESState bit = (s >> src) & 1;
        t |= bit << dst;
    }
    return t;
}

__device__ ESState applyMSMoveGPU(ESState s, int m) {
    return applySliceMoveGPU(s, m);
}
__device__ ESState applySSliceMoveGPU(ESState s, int m) {
    return applySliceMoveGPU(s, m);
}

// triad‑membership test: each corner slot and its piece both lie in {0,3,4,7} or both in {1,2,5,6}
__device__ bool inTriads(CPState cp) {
    for (int slot = 0; slot < 8; ++slot) {
        int piece = (cp >> (3*slot)) & 0x7;
        bool slotA  = (slot==0||slot==3||slot==4||slot==7);
        bool pieceA = (piece==0||piece==3||piece==4||piece==7);
        if (slotA != pieceA) return false;
    }
    return true;
}

// brute‑force kernel
__global__ void bruteForceStage3Kernel(
    CPState    startCP,
    ESState    startM,
    ESState    startS,
    int        depth,
    int       *solutionBuffer,
    int       *solutionFoundFlag
) {
    unsigned long long tid = blockIdx.x*blockDim.x + threadIdx.x;
    // total = 14^depth
    unsigned long long total = 1;
    for (int i = 0; i < depth; ++i) total *= 14ULL;
    if (tid >= total) return;

    CPState cp    = startCP;
    ESState mMask = startM;
    ESState sMask = startS;
    int seq[MAX_STAGE3_DEPTH];
    unsigned long long code = tid;
    int prevFace = -1;

    for (int d = 0; d < depth; ++d) {
        int sel  = code % 14;  code /= 14;
        int m    = allowedMoves[sel];
        int face = m / 3;
        if (face == prevFace) return;
        prevFace = face;

        seq[d]   = m;
        cp       = applyCPMoveGPU(cp, m);
        mMask    = applyMSMoveGPU(mMask, m);
        sMask    = applySSliceMoveGPU(sMask, m);
    }

    // check M‑slice, S‑slice, and triad‑membership
    if ((mMask & targetMSliceMask) == targetMSliceMask &&
        (sMask & targetSSliceMask) == targetSSliceMask &&
         inTriads(cp))
    {
        if (atomicExch(solutionFoundFlag, 1) == 0) {
            for (int i = 0; i < depth; ++i)
                solutionBuffer[i] = seq[i];
        }
    }
}

std::vector<std::string> solveStage3(u64 cornerState, u64 edgeState) {
    // 1) pack corner‑permutation into 3 bits/slot
    CPState startCP = 0;
    for (int i = 0; i < 8; ++i) {
        int piece = (cornerState >> (5*i)) & 0x7;
        startCP  |= CPState(piece) << (3*i);
    }

    // 2) pack M‑slice & S‑slice membership
    ESState startM = 0, startS = 0;
    for (int i = 0; i < 12; ++i) {
        int piece = (edgeState >> (5*i)) & 0x1F;
        bool inM   = (piece==0||piece==2||piece==4||piece==6);
        bool inS   = (piece==1||piece==3||piece==5||piece==7);
        startM |= ESState(inM) << i;
        startS |= ESState(inS) << i;
    }

    // 3) alloc GPU buffers
    int *d_sol, *d_flag;
    cudaMalloc(&d_sol,  sizeof(int)*MAX_STAGE3_DEPTH);
    cudaMalloc(&d_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_flag, &zero, sizeof(zero), cudaMemcpyHostToDevice);

    // 4) iterative‑deepening search
    int hostSol[MAX_STAGE3_DEPTH], foundDepth = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int depth = 1; depth <= MAX_STAGE3_DEPTH; ++depth) {
        unsigned long long total = 1;
        for (int i = 0; i < depth; ++i) total *= 14ULL;
        int threads = 256;
        int blocks  = int((total + threads - 1)/threads);

        bruteForceStage3Kernel<<<blocks,threads>>>(
            startCP, startM, startS, depth,
            d_sol, d_flag
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
    std::cout << "Stage 3 solved in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
              << " ms\n";

    cudaFree(d_sol);
    cudaFree(d_flag);

    // 5) map move‑indices back to notation
    static const char* names[18] = {
      "U","U2","U'", "R","R2","R'", "F","F2","F'",
      "D","D2","D'", "L","L2","L'", "B","B2","B'"
    };
    std::vector<std::string> sol;
    sol.reserve(foundDepth);
    for (int i = 0; i < foundDepth; ++i)
        sol.emplace_back(names[hostSol[i]]);
    return sol;
}
