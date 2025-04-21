// stage3.cu
// Thistlethwaite Stage 3 solver with precomputed bitboard transitions

#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>   // for memset

using u64 = uint64_t;

// —— CONFIG ——
#define MAX_SEARCH_DEPTH_STAGE3 11
#define NUM_MOVES_STAGE3 10
#define THREADS_PER_BLOCK 256

// target masks for G₃ (half‑turn subgroup)
static const uint16_t TARGET_SLICE_MASK  = (1<<8)|(1<<9)|(1<<10)|(1<<11);
static const uint8_t  TARGET_TETRAD_MASK = (1<<0)|(1<<3)|(1<<4)|(1<<7);

// notation names
static const char* moveNotationNames[18] = {
    "U","U2","U'",
    "R","R2","R'",
    "F","F2","F'",
    "D","D2","D'",
    "L","L2","L'",
    "B","B2","B'"
};

// —— DEVICE CONSTANTS ——

// edge‐perm table (from Stage1)
__device__ __constant__ int eoPerm[18][12] = {
    {3,0,1,2,4,5,6,7,8,9,10,11},
    {2,3,0,1,4,5,6,7,8,9,10,11},
    {1,2,3,0,4,5,6,7,8,9,10,11},
    {8,1,2,3,11,5,6,7,4,9,10,0},
    {4,1,2,3,0,5,6,7,11,9,10,8},
    {11,1,2,3,8,5,6,7,0,9,10,4},
    {0,9,2,3,4,8,6,7,1,5,10,11},
    {0,5,2,3,4,1,6,7,9,8,10,11},
    {0,8,2,3,4,9,6,7,5,1,10,11},
    {0,1,2,3,5,6,7,4,8,9,10,11},
    {0,1,2,3,6,7,4,5,8,9,10,11},
    {0,1,2,3,7,4,5,6,8,9,10,11},
    {0,1,10,3,4,5,9,7,8,2,6,11},
    {0,1,6,3,4,5,2,7,8,10,9,11},
    {0,1,9,3,4,5,10,7,8,6,2,11},
    {0,1,2,11,4,5,6,10,8,9,3,7},
    {0,1,2,7,4,5,6,3,8,9,11,10},
    {0,1,2,10,4,5,6,11,8,9,7,3}
};

// corner‐perm table (from Stage2)
__device__ __constant__ int coPerm[18][8] = {
    {3,0,1,2,4,5,6,7},{2,3,0,1,4,5,6,7},{1,2,3,0,4,5,6,7},
    {4,1,2,0,7,5,6,3},{7,1,2,4,3,5,6,0},{3,1,2,7,0,5,6,4},
    {1,5,2,3,0,4,6,7},{5,4,2,3,1,0,6,7},{4,0,2,3,5,1,6,7},
    {0,1,2,3,5,6,7,4},{0,1,2,3,6,7,4,5},{0,1,2,3,7,4,5,6},
    {0,2,6,3,4,1,5,7},{0,6,5,3,4,2,1,7},{0,5,1,3,4,6,2,7},
    {0,1,3,7,4,5,2,6},{0,1,7,6,4,5,3,2},{0,1,6,2,4,5,7,3}
};

// allowed moves: U,U2,U', D,D2,D', R2, F2, L2, B2
__device__ __constant__ int allowedMoves3[NUM_MOVES_STAGE3] = {
    0, 1, 2,    // U, U2, U'
    9,10,11,    // D, D2, D'
    4,          // R2
    7,          // F2
    13,         // L2
    16          // B2
};

// precomputed transition tables
__device__ uint16_t d_sliceTrans[NUM_MOVES_STAGE3][4096];
__device__ uint8_t  d_tetraTrans[NUM_MOVES_STAGE3][256];

// —— HOST COPIES FOR TABLE BUILDING ——
static const int h_eoPerm[18][12] = {
    {3,0,1,2,4,5,6,7,8,9,10,11},
    {2,3,0,1,4,5,6,7,8,9,10,11},
    {1,2,3,0,4,5,6,7,8,9,10,11},
    {8,1,2,3,11,5,6,7,4,9,10,0},
    {4,1,2,3,0,5,6,7,11,9,10,8},
    {11,1,2,3,8,5,6,7,0,9,10,4},
    {0,9,2,3,4,8,6,7,1,5,10,11},
    {0,5,2,3,4,1,6,7,9,8,10,11},
    {0,8,2,3,4,9,6,7,5,1,10,11},
    {0,1,2,3,5,6,7,4,8,9,10,11},
    {0,1,2,3,6,7,4,5,8,9,10,11},
    {0,1,2,3,7,4,5,6,8,9,10,11},
    {0,1,10,3,4,5,9,7,8,2,6,11},
    {0,1,6,3,4,5,2,7,8,10,9,11},
    {0,1,9,3,4,5,10,7,8,6,2,11},
    {0,1,2,11,4,5,6,10,8,9,3,7},
    {0,1,2,7,4,5,6,3,8,9,11,10},
    {0,1,2,10,4,5,6,11,8,9,7,3}
};

static const int h_coPerm[18][8] = {
    {3,0,1,2,4,5,6,7},{2,3,0,1,4,5,6,7},{1,2,3,0,4,5,6,7},
    {4,1,2,0,7,5,6,3},{7,1,2,4,3,5,6,0},{3,1,2,7,0,5,6,4},
    {1,5,2,3,0,4,6,7},{5,4,2,3,1,0,6,7},{4,0,2,3,5,1,6,7},
    {0,1,2,3,5,6,7,4},{0,1,2,3,6,7,4,5},{0,1,2,3,7,4,5,6},
    {0,2,6,3,4,1,5,7},{0,6,5,3,4,2,1,7},{0,5,1,3,4,6,2,7},
    {0,1,3,7,4,5,2,6},{0,1,7,6,4,5,3,2},{0,1,6,2,4,5,7,3}
};

static const int h_allowedMoves3[NUM_MOVES_STAGE3] = {
    0,1,2, 9,10,11, 4,7,13,16
};

// —— BUILD & UPLOAD TABLES ——
void buildAndUploadTransTables() {
    static uint16_t h_sliceTrans[NUM_MOVES_STAGE3][4096];
    static uint8_t  h_tetraTrans[NUM_MOVES_STAGE3][256];
    memset(h_sliceTrans,  0, sizeof(h_sliceTrans));
    memset(h_tetraTrans,  0, sizeof(h_tetraTrans));

    // for each allowed move index
    for(int mIdx = 0; mIdx < NUM_MOVES_STAGE3; ++mIdx) {
        int m = h_allowedMoves3[mIdx];
        // slice transitions:
        for(int mask = 0; mask < 4096; ++mask) {
            uint16_t out = 0;
            for(int dst = 0; dst < 12; ++dst) {
                int src = h_eoPerm[m][dst];
                if (mask & (1u<<src)) out |= 1u<<dst;
            }
            h_sliceTrans[mIdx][mask] = out;
        }
        // tetrad transitions:
        for(int mask = 0; mask < 256; ++mask) {
            uint8_t out = 0;
            for(int src = 0; src < 8; ++src) {
                if (mask & (1u<<src)) {
                    int dst = h_coPerm[m][src];
                    out |= 1u<<dst;
                }
            }
            h_tetraTrans[mIdx][mask] = out;
        }
    }

    // copy into device constant memory
    cudaMemcpyToSymbol(d_sliceTrans, h_sliceTrans,
                       sizeof(h_sliceTrans));
    cudaMemcpyToSymbol(d_tetraTrans, h_tetraTrans,
                       sizeof(h_tetraTrans));
}

// —— SOLVING KERNEL ——
__global__ void bruteForceStage3Kernel(
    uint16_t startSliceMask,
    uint8_t  startTetradMask,
    int      depth,
    int     *solutionBuffer,
    int     *solutionFoundFlag
) {
    unsigned long long tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    // total sequences = NUM_MOVES3^depth
    unsigned long long total = 1;
    for(int i = 0; i < depth; ++i) total *= NUM_MOVES_STAGE3;
    if (tid >= total) return;

    uint16_t sliceMask  = startSliceMask;
    uint8_t  tetradMask = startTetradMask;
    int      prevFace   = -1;
    int      seq[MAX_SEARCH_DEPTH_STAGE3];
    unsigned long long code = tid;

    // decode & update via two table lookups
    for(int d = 0; d < depth; ++d) {
        int sel = int(code % NUM_MOVES_STAGE3);
        code   /= NUM_MOVES_STAGE3;

        int m = allowedMoves3[sel];
        int face = m / 3;
        if (face == prevFace) return;
        prevFace = face;
        seq[d] = m;

        // table lookups
        sliceMask  = d_sliceTrans[sel][sliceMask];
        tetradMask = d_tetraTrans[sel][tetradMask];
    }

    // check goal
    if (sliceMask == TARGET_SLICE_MASK &&
        tetradMask == TARGET_TETRAD_MASK)
    {
        if (atomicExch(solutionFoundFlag, 1) == 0) {
            for(int i = 0; i < depth; ++i)
                solutionBuffer[i] = seq[i];
        }
    }
}

// —— HOST SOLVER ——
std::vector<std::string> solveStage3(u64 cornerState, u64 edgeState) {
    // 1) build start‐masks
    uint16_t sliceMask = 0;
    for(int slot = 0; slot < 12; ++slot) {
        int packed = (edgeState >> (5*slot)) & 0x1F;
        int piece  = packed & 0xF;
        if (piece >= 8 && piece <= 11)
            sliceMask |= 1u<<slot;
    }
    uint8_t tetradMask = 0;
    for(int slot = 0; slot < 8; ++slot) {
        int packed = (cornerState >> (5*slot)) & 0x1F;
        int piece  = packed & 0x7;
        if (piece == 0 || piece == 3 || piece == 4 || piece == 7)
            tetradMask |= 1u<<slot;
    }
    // already in G₃?
    if (sliceMask == TARGET_SLICE_MASK &&
        tetradMask == TARGET_TETRAD_MASK)
        return {};

    // 2) build & upload our transition tables
    buildAndUploadTransTables();

    // 3) allocate device buffers
    int *d_sol, *d_flag;
    cudaMalloc(&d_sol,  sizeof(int)*MAX_SEARCH_DEPTH_STAGE3);
    cudaMalloc(&d_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_flag, &zero, sizeof(zero), cudaMemcpyHostToDevice);

    // 4) iterative deepening
    int hostSol[MAX_SEARCH_DEPTH_STAGE3];
    int foundDepth = 0;
    auto t0 = std::chrono::steady_clock::now();
    for(int depth = 1; depth <= MAX_SEARCH_DEPTH_STAGE3; ++depth) {
        // compute total = NUM_MOVES3^depth
        unsigned long long total = 1;
        for(int i = 0; i < depth; ++i) total *= NUM_MOVES_STAGE3;
        int blocks = int((total + THREADS_PER_BLOCK - 1)
                         / THREADS_PER_BLOCK);

        bruteForceStage3Kernel
          <<<blocks,THREADS_PER_BLOCK>>>(sliceMask,
                                         tetradMask,
                                         depth,
                                         d_sol,
                                         d_flag);
        cudaDeviceSynchronize();

        int found = 0;
        cudaMemcpy(&found, d_flag,
                   sizeof(found),
                   cudaMemcpyDeviceToHost);
        if (found) {
            foundDepth = depth;
            cudaMemcpy(hostSol, d_sol,
                       sizeof(int)*depth,
                       cudaMemcpyDeviceToHost);
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Stage 3 solved in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     t1 - t0).count()
              << " ms\n";

    // 5) cleanup
    cudaFree(d_sol);
    cudaFree(d_flag);

    // 6) translate to notation
    std::vector<std::string> res;
    res.reserve(foundDepth);
    for(int i = 0; i < foundDepth; ++i)
        res.emplace_back(moveNotationNames[hostSol[i]]);
    return res;
}
