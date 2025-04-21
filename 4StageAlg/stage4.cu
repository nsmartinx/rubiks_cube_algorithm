#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

using u64     = uint64_t;
// 8 corners ×3 bits each = 24 bits
using CPState = uint32_t;
// 12 edges  ×4 bits each = 48 bits (fits in u64)
using EPState = uint64_t;

#define MAX_STAGE4_DEPTH 15

// —–– only double‑turn moves (indices into your 18‑move table) ——
__device__ __constant__ int allowedMoves4[6] = {
    1,   // U2
    4,   // R2
    7,   // F2
    10,  // D2
    13,  // L2
    16   // B2
};

// —–– the *true* solved‐state labels for comparison ——
__device__ __constant__ CPState ID_CP = 0x00FAC688u;          // ∑ i<<(3*i), i=0..7
__device__ __constant__ EPState ID_EP = 0x0BA9876543210ULL;  // ∑ i<<(4*i), i=0..11

// —–– dst→src tables, copied verbatim from before ——
__device__ __constant__ int coPerm[18][8] = {
    {3,0,1,2,4,5,6,7},{2,3,0,1,4,5,6,7},{1,2,3,0,4,5,6,7},
    {4,1,2,0,7,5,6,3},{7,1,2,4,3,5,6,0},{3,1,2,7,0,5,6,4},
    {1,5,2,3,0,4,6,7},{5,4,2,3,1,0,6,7},{4,0,2,3,5,1,6,7},
    {0,1,2,3,5,6,7,4},{0,1,2,3,6,7,4,5},{0,1,2,3,7,4,5,6},
    {0,2,6,3,4,1,5,7},{0,6,5,3,4,2,1,7},{0,5,1,3,4,6,2,7},
    {0,1,3,7,4,5,2,6},{0,1,7,6,4,5,3,2},{0,1,6,2,4,5,7,3}
};
__device__ __constant__ int ePerm[18][12] = {
  {3,0,1,2,4,5,6,7,8,9,10,11},{2,3,0,1,4,5,6,7,8,9,10,11},{1,2,3,0,4,5,6,7,8,9,10,11},
  {8,1,2,3,11,5,6,7,4,9,10,0},{4,1,2,3,0,5,6,7,11,9,10,8},{11,1,2,3,8,5,6,7,0,9,10,4},
  {0,9,2,3,4,8,6,7,1,5,10,11},{0,5,2,3,4,1,6,7,9,8,10,11},{0,8,2,3,4,9,6,7,5,1,10,11},
  {0,1,2,3,5,6,7,4,8,9,10,11},{0,1,2,3,6,7,4,5,8,9,10,11},{0,1,2,3,7,4,5,6,8,9,10,11},
  {0,1,10,3,4,5,9,7,8,2,6,11},{0,1,6,3,4,5,2,7,8,10,9,11},{0,1,9,3,4,5,10,7,8,6,2,11},
  {0,1,2,11,4,5,6,10,8,9,3,7},{0,1,2,7,4,5,6,3,8,9,11,10},{0,1,2,10,4,5,6,11,8,9,7,3}
};

// —–– apply corner‑permutation (just permute raw piece‐labels) ——
__device__ CPState applyCPMoveGPU2(CPState s, int m) {
    CPState t = 0;
    for (int dst = 0; dst < 8; ++dst) {
        int src   = coPerm[m][dst];
        CPState c = (s >> (3*src)) & 0x7;
        t |= c << (3*dst);
    }
    return t;
}

// —–– apply edge‑permutation (raw labels 0..11, 4 bits each) ——
__device__ EPState applyEPMoveGPU(EPState s, int m) {
    EPState t = 0;
    for (int dst = 0; dst < 12; ++dst) {
        int src   = ePerm[m][dst];
        EPState e = (s >> (4*src)) & 0xF;
        t |= e << (4*dst);
    }
    return t;
}

// —–– brute‑force kernel for stage 4 ——
__global__ void bruteForceStage4Kernel(
    CPState    startCP,
    EPState    startEP,
    int        depth,
    int       *solutionBuffer,
    int       *solutionFoundFlag
) {
    unsigned long long tid =
      blockIdx.x * blockDim.x + threadIdx.x;

    // total = 6^depth
    unsigned long long total = 1;
    for (int i = 0; i < depth; ++i) total *= 6ULL;
    if (tid >= total) return;

    CPState cp = startCP;
    EPState ep = startEP;
    int seq[MAX_STAGE4_DEPTH];
    unsigned long long code = tid;
    int prevFace = -1;

    // build the move sequence
    for (int d = 0; d < depth; ++d) {
        int sel  = code % 6;  code /= 6;
        int m    = allowedMoves4[sel];
        int face = m / 3;
        if (face == prevFace) return;
        prevFace = face;
        seq[d] = m;
        cp = applyCPMoveGPU2(cp, m);
        ep = applyEPMoveGPU(ep, m);
    }

    // **now** solved when cp==ID_CP and ep==ID_EP
    if (cp == ID_CP && ep == ID_EP) {
        if (atomicExch(solutionFoundFlag, 1) == 0) {
            for (int i = 0; i < depth; ++i)
                solutionBuffer[i] = seq[i];
        }
    }
}

// —–– packers: record *raw* piece labels so solved→ID_CP/ID_EP ——
CPState packCP(u64 cornerState) {
    CPState s = 0;
    for (int i = 0; i < 8; ++i) {
        int piece = (cornerState >> (5*i)) & 0x7;  // 0..7
        s |= CPState(piece) << (3*i);
    }
    return s;
}

EPState packEP(u64 edgeState) {
    EPState s = 0;
    for (int i = 0; i < 12; ++i) {
        int piece = (edgeState >> (5*i)) & 0x1F;   // 0..11
        s |= EPState(piece & 0xF) << (4*i);
    }
    return s;
}

std::vector<std::string> solveStage4(u64 cornerState, u64 edgeState) {
    // 1) pack so that solved→ID_CP/ID_EP
    CPState startCP = packCP(cornerState);
    EPState startEP = packEP(edgeState);

    // 2) allocate GPU buffers
    int *d_sol, *d_flag;
    cudaMalloc(&d_sol,  sizeof(int)*MAX_STAGE4_DEPTH);
    cudaMalloc(&d_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_flag, &zero, sizeof(zero), cudaMemcpyHostToDevice);

    // 3) iterative deepening
    int hostSol[MAX_STAGE4_DEPTH], foundDepth = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int depth = 1; depth <= MAX_STAGE4_DEPTH; ++depth) {
        unsigned long long total = 1;
        for (int i = 0; i < depth; ++i) total *= 6ULL;
        int threads = 256;
        int blocks  = int((total + threads - 1) / threads);

        bruteForceStage4Kernel<<<blocks,threads>>>(
            startCP, startEP, depth,
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
    std::cout << "Stage 4 (double‑turn solve) in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    cudaFree(d_sol);
    cudaFree(d_flag);

    // 4) translate to notation and return
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
