// stage1.cu
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <bitset>

using u64     = uint64_t;
using EOState = uint16_t;

#define MAX_DEPTH_EO 7

// Edge permutation
__device__ __constant__ int eoPerm[18][12] = {
    /* U  */ {3,0,1,2,4,5,6,7,8,9,10,11},
    /* U2 */ {2,3,0,1,4,5,6,7,8,9,10,11},
    /* U' */ {1,2,3,0,4,5,6,7,8,9,10,11},

    /* R  */ {8,1,2,3,11,5,6,7,4,9,10,0},
    /* R2 */ {4,1,2,3,0,5,6,7,11,9,10,8},
    /* R' */ {11,1,2,3,8,5,6,7,0,9,10,4},

    /* F  */ {0,9,2,3,4,8,6,7,1,5,10,11},
    /* F2 */ {0,5,2,3,4,1,6,7,9,8,10,11},
    /* F' */ {0,8,2,3,4,9,6,7,5,1,10,11},

    /* D  */ {0,1,2,3,5,6,7,4,8,9,10,11},
    /* D2 */ {0,1,2,3,6,7,4,5,8,9,10,11},
    /* D' */ {0,1,2,3,7,4,5,6,8,9,10,11},

    /* L  */ {0,1,10,3,4,5,9,7,8,2,6,11},
    /* L2 */ {0,1,6,3,4,5,2,7,8,10,9,11},
    /* L' */ {0,1,9,3,4,5,10,7,8,6,2,11},

    /* B  */ {0,1,2,11,4,5,6,10,8,9,3,7},
    /* B2 */ {0,1,2,7,4,5,6,3,8,9,11,10},
    /* B' */ {0,1,2,10,4,5,6,11,8,9,7,3}
};

// Flip‑mask
__device__ __constant__ EOState eoFlipMask[18] = {
    /* U  */ 0b0000'0000'0000,
    /* U2 */ 0b0000'0000'0000,
    /* U' */ 0b0000'0000'0000,

    /* R  */ 0b0000'0000'0000,
    /* R2 */ 0b0000'0000'0000,
    /* R' */ 0b0000'0000'0000,

    /* F  */ 0b0011'0010'0010,
    /* F2 */ 0b0000'0000'0000,
    /* F' */ 0b0011'0010'0010,

    /* D  */ 0b0000'0000'0000,
    /* D2 */ 0b0000'0000'0000,
    /* D' */ 0b0000'0000'0000,

    /* L  */ 0b0000'0000'0000,
    /* L2 */ 0b0000'0000'0000,
    /* L' */ 0b0000'0000'0000,

    /* B  */ 0b1100'1000'1000,
    /* B2 */ 0b0000'0000'0000,
    /* B' */ 0b1100'1000'1000
};

// For printing
static const char* moveNotationNames[18] = {
    "U","U2","U'",
    "R","R2","R'",
    "F","F2","F'",
    "D","D2","D'",
    "L","L2","L'",
    "B","B2","B'"
};

// Apply one move
__device__ EOState applyEOMoveGPU(EOState s, int m) {
    EOState t = 0, flip = eoFlipMask[m];
    for (int dst = 0; dst < 12; ++dst) {
        int src = eoPerm[m][dst];
        int ori = (s >> src) & 1;
        if ((flip >> src) & 1) ori ^= 1;
        t |= (EOState(ori) << dst);
    }
    return t;
}

// Kernel: brute‑force all sequences up to depth, skipping same‑face adjacencies
__global__ void bruteForceEOKernel(
    EOState startState,
    int depth,
    int *solutionBuffer,
    int *solutionFlag
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    // total = 18^depth
    unsigned long long total = 1;
    for (int i = 0; i < depth; ++i) total *= 18ULL;
    if (tid >= total) return;

    EOState state = startState;
    int seq[MAX_DEPTH_EO];
    unsigned long long code = tid;
    int prevFace = -1;

    // decode & apply
    for (int d = 0; d < depth; ++d) {
        int m = code % 18; code /= 18;
        int face = m / 3;
        if (face == prevFace) return;
        prevFace = face;
        seq[d] = m;
        state = applyEOMoveGPU(state, m);
    }

    if (state == 0) {
        if (atomicExch(solutionFlag, 1) == 0) {
            for (int i = 0; i < depth; ++i)
                solutionBuffer[i] = seq[i];
        }
    }
}

std::vector<std::string> solveStage1(u64 edgeState) {
    // Pack EO bits
    EOState eo = 0;
    for (int i = 0; i < 12; ++i) {
        int packed = (edgeState >> (5*i)) & 0x1F;
        int ori    = (packed >> 4) & 1;
        eo |= EOState(ori << i);
    }

    if (eo == 0) return {};

    // alloc
    int *d_sol, *d_flag;
    cudaMalloc(&d_sol,  sizeof(int)*MAX_DEPTH_EO);
    cudaMalloc(&d_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_flag, &zero, sizeof(zero), cudaMemcpyHostToDevice);

    // brute‑force
    auto t0 = std::chrono::steady_clock::now();
    int hostSol[MAX_DEPTH_EO];
    int foundDepth = 0;
    for (int depth = 1; depth <= MAX_DEPTH_EO; ++depth) {
        unsigned long long total = 1;
        for (int i = 0; i < depth; ++i) total *= 18ULL;
        int tpB = 256;
        int blocks = (total + tpB - 1)/tpB;

        bruteForceEOKernel<<<blocks,tpB>>>(eo, depth, d_sol, d_flag);
        cudaDeviceSynchronize();

        int found;
        cudaMemcpy(&found, d_flag, sizeof(found), cudaMemcpyDeviceToHost);
        if (found) {
            foundDepth = depth;
            cudaMemcpy(hostSol, d_sol, sizeof(int)*depth, cudaMemcpyDeviceToHost);
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Stage 1 (EO) solved in " << ms << " ms\n";

    cudaFree(d_sol);
    cudaFree(d_flag);

    // build result
    std::vector<std::string> res;
    res.reserve(foundDepth);
    for (int i = 0; i < foundDepth; ++i)
        res.emplace_back(moveNotationNames[hostSol[i]]);
    return res;
}
