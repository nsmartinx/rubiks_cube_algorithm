#include <iostream>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

using u64 = uint64_t;

// —— Configuration ——
#define MAX_DEPTH 8
#define THREADS_PER_BLOCK 256

// —— Cubie‐level raw quarter‑turn moves ——
// corner slots: 0=URF,1=UFL,2=ULB,3=UBR,4=DFR,5=DLF,6=DBL,7=DRB
// edge   slots: 0=UR, 1=UF, 2=UL, 3=UB, 4=DR, 5=DF, 6=DL, 7=DB, 8=FR, 9=FL, 10=BL, 11=BR
struct RawMove {
    int cp[8], co[8];
    int ep[12], eo[12];
};

// host‐side definitions
static const RawMove h_quarterMoves[6] = {
    // U
    { {3,0,1,2,4,5,6,7}, {0,0,0,0,0,0,0,0},
      {3,0,1,2,4,5,6,7,8,9,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} },
    // R
    { {4,1,2,0,7,5,6,3}, {2,0,0,1,1,0,0,2},
      {8,1,2,3,11,5,6,7,4,9,10,0}, {0,0,0,0,0,0,0,0,0,0,0,0} },
    // F
    { {1,5,2,3,0,4,6,7}, {1,2,0,0,2,1,0,0},
      {0,9,2,3,4,8,6,7,1,5,10,11}, {0,1,0,0,0,1,0,0,1,1,0,0} },
    // D
    { {0,1,2,3,5,6,7,4}, {0,0,0,0,0,0,0,0},
      {0,1,2,3,5,6,7,4,8,9,10,11}, {0,0,0,0,0,0,0,0,0,0,0,0} },
    // L
    { {0,2,6,3,4,1,5,7}, {0,1,2,0,0,2,1,0},
      {0,1,10,3,4,5,9,7,8,2,6,11}, {0,0,0,0,0,0,0,0,0,0,0,0} },
    // B
    { {0,1,3,7,4,5,2,6}, {0,0,1,2,0,0,2,1},
      {0,1,2,11,4,5,6,10,8,9,3,7}, {0,0,0,1,0,0,0,1,0,0,1,1} }
};

// device‐side (constant) copies
__constant__ RawMove d_quarterMoves[6];
__constant__ u64 d_solvedC;
__constant__ u64 d_solvedE;

// apply one quarter‐turn on GPU (in‐place)
__device__ inline void applyQuarterMoveGPU(u64 &C, u64 &E, int f) {
    const RawMove &m = d_quarterMoves[f];
    u64 oldC = C, oldE = E;
    u64 newC = 0, newE = 0;

    // corners
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        u64 field = (oldC >> (5*i)) & 0x1FULL;
        int idx =  field        & 0x7;        // bits [0..2]
        int ori = (field >> 3)  & 0x3;        // bits [3..4]
        int newOri = (ori + m.co[i]) % 3;
        int dest   = m.cp[i];
        u64 nf = u64(idx) | (u64(newOri) << 3);
        newC |= (nf << (5*dest));
    }

    // edges
    #pragma unroll
    for (int j = 0; j < 12; ++j) {
        u64 field = (oldE >> (5*j)) & 0x1FULL;
        int idx =   field        & 0xF;       // bits [0..3]
        int ori =  (field >> 4)  & 0x1;       // bit 4
        int newOri = ori ^ m.eo[j];
        int dest   = m.ep[j];
        u64 nf = u64(idx) | (u64(newOri) << 4);
        newE |= (nf << (5*dest));
    }

    C = newC;
    E = newE;
}

// GPU kernel: each thread decodes a unique sequence of length `depth` and applies it
__global__ void bruteKernel(int depth, u64 totalSequences) {
    u64 idx = u64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= totalSequences) return;

    // start from solved
    u64 C = d_solvedC;
    u64 E = d_solvedE;

    // decode idx in base‑18, LS digit first
    u64 tmp = idx;
    for (int i = 0; i < depth; ++i) {
        int m = tmp % 18;
        tmp /= 18;
        int face = m / 3;
        int turn = m % 3;
        int times = (turn == 1 ? 2 : (turn == 2 ? 3 : 1));
        #pragma unroll
        for (int t = 0; t < times; ++t) {
            applyQuarterMoveGPU(C, E, face);
        }
    }
    // we don't store C/E anywhere — this is purely to exercise the bit‑ops
}

int main() {
    // copy moves into constant memory
    cudaMemcpyToSymbol(d_quarterMoves, h_quarterMoves, sizeof(h_quarterMoves));
    // build and copy solved‐state
    u64 solvedC = 0, solvedE = 0;
    for (int i = 0; i < 8; ++i) solvedC |= (u64(i) << (5*i));
    for (int j = 0; j < 12; ++j) solvedE |= (u64(j) << (5*j));
    cudaMemcpyToSymbol(d_solvedC, &solvedC, sizeof(u64));
    cudaMemcpyToSymbol(d_solvedE, &solvedE, sizeof(u64));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int depth = 1; depth <= MAX_DEPTH; ++depth) {
        // total number of sequences = 18^depth
        u64 totalSeq = 1;
        for (int i = 0; i < depth; ++i) totalSeq *= 18;

        // launch configuration
        u64 blocks = (totalSeq + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // time the kernel
        cudaEventRecord(start);
        bruteKernel<<<blocks, THREADS_PER_BLOCK>>>(depth, totalSeq);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Depth " << depth
                  << "completed in " << ms << " ms\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
