#include "solver_kernel.h"
#include "cube_defs.h"
#include <vector>
#include <string>
#include <cstdint>

__device__ __constant__ int d_allowedMovesStage1[18] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
__device__ __constant__ int d_allowedMovesStage2[14] = {0,1,2,3,4,5,7,9,10,11,12,13,14,16};
__device__ __constant__ int d_allowedMovesStage3[10] = {0,1,2,4,7,9,10,11,13,16};
__device__ __constant__ int d_allowedMovesStage4[6] = {1,4,7,10,13,16};

// these globals are set in main.cpp before each stage
extern u64 g_rawEdgeState;
extern u64 g_rawCornerState;

//=== Stage 1: edge‐orientation only (18 moves, MaxDepth=7) ===//

struct ApplyEO {
    static __device__ void apply(u16 &state, int move) {
        state = applyEdgeFlip(state, move);
    }
};
struct IsSolved1 {
    static __device__ bool check(u16 state) {
        return state == 0;
    }
};
u16 packEO() {
    u16 packed = 0;
    for (int i = 0; i < 12; ++i) {
        packed |= u16((g_rawEdgeState >> (5*i + 4) & 1u) << i);
    }
    return packed;
}

std::vector<std::string> solveStage1() {
    const int* d_moves1 = nullptr;
    cudaGetSymbolAddress((void**)&d_moves1, d_allowedMovesStage1);
    return solveStage<18, u16, ApplyEO, IsSolved1, 7>(packEO, 7, d_moves1);
}

//=== Stage 2: corner‐twist + E‐slice (14 moves, MaxDepth=10) ===//

// We'll pack into a 32‐bit: low 16 bits corner‐twist (2 bits each), high 16 bits E‐slice mask (1 bit each)
struct PackStage2 {
    static u32 pack() {
        u16 twist = 0;
        for (int i = 0; i < 8; ++i) {
            u32 slot = (g_rawCornerState >> (5*i)) & 0x1Fu;
            u32 ori  = (slot >> 3) & 0x3u;
            twist |= u16(ori) << (2*i);
        }
        u16 slice = 0;
        for (int i = 0; i < 12; ++i) {
            u32 piece = (g_rawEdgeState >> (5*i)) & 0x1Fu;
            bool inE = (piece >=  8 && piece <= 11);
            slice |= u16(inE) << i;
        }
        return u32(twist) | (u32(slice) << 16);
    }
};

struct ApplyStage2 {
    static __device__ void apply(u32 &state, int move) {
        u16 twist = state & 0xFFFFu;
        u16 slice = state >> 16;
        twist = applyCornerTwistMoveGpu(twist, move);
        slice = applyEdgePerm(slice, move);
        state = u32(twist) | (u32(slice) << 16);
    }
};

struct IsSolved2 {
    static __device__ bool check(u32 state) {
        u16 twist = state & 0xFFFFu;
        u16 slice = state >> 16;
        return twist == 0
            && (slice & d_equatorSliceMask) == d_equatorSliceMask;
    }
};



std::vector<std::string> solveStage2() {
    const int* d_moves2 = nullptr;
    cudaGetSymbolAddress((void**)&d_moves2, d_allowedMovesStage2);
    return solveStage<14, u32, ApplyStage2, IsSolved2, 10>(
        PackStage2::pack, 10, d_moves2
    );
}

//=== Stage 3: double‐move reduction (10 moves, MaxDepth=13) ===//

// Pack 3‐bit corner perm + 12‐bit middle‐slice mask into a u64
struct PackStage3 {
    static u64 pack() {
        u32 cornerPack = 0;
        for (int i = 0; i < 8; ++i) {
            u32 piece = (g_rawCornerState >> (5*i)) & 7u;
            cornerPack |= piece << (3*i);
        }
        u16 sliceMask = 0;
        for (int i = 0; i < 12; ++i) {
            u32 packed = (g_rawEdgeState >> (5*i)) & 0x1Fu;
            u32 piece  = packed & 0xFu;
            bool inMiddle = (piece == 0 || piece == 2 || piece == 4 || piece == 6);
            sliceMask |= u16(inMiddle) << i;
        }
        return u64(cornerPack) | (u64(sliceMask) << 32);
    }
};

struct ApplyStage3 {
    static __device__ void apply(u64 &state, int move) {
        u32 corners = u32(state);
        u16 slice   = u16(state >> 32);
        corners = applyCornerPerm(corners, move);
        slice   = applyEdgePerm(slice,   move);
        state = u64(corners) | (u64(slice) << 32);
    }
};

struct IsSolved3 {
    static __device__ bool check(u64 state) {
        u32 corners = u32(state);
        u16 slice   = u16(state >> 32);
        // check middle slice
        if ((slice & d_middleSliceMask) != d_middleSliceMask) return false;
        // parity
        int arr[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) arr[i] = (corners >> (3*i)) & 7;
        int invParity = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            for (int j = i+1; j < 8; ++j)
                invParity ^= (arr[i] > arr[j]);
        if (invParity != 0) return false;
        // pairing groups
        unsigned maskAccum = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int piece = (corners >> (3*i)) & 7;
            maskAccum |= ((piece ^ i) & 5u);
        }
        return maskAccum == 0;
    }
};


std::vector<std::string> solveStage3() {
    const int* d_moves3 = nullptr;
    cudaGetSymbolAddress((void**)&d_moves3, d_allowedMovesStage3);
    return solveStage<10, u64, ApplyStage3, IsSolved3, 13>(
        PackStage3::pack, 13, d_moves3
    );
}

//=== Stage 4: final double‐turn solve (6 moves, MaxDepth=15) ===//

// state struct for separate corner and full-edge perms
struct Stage4State {
    u32 cornerState;
    u64 edgeState;
};

static constexpr u32 IDENTITY_CORNER = 16434824u;          // sum(i<<(3*i)) for i=0..7
static constexpr u64 IDENTITY_EDGE   = 205163983024656ull; // sum(i<<(4*i)) for i=0..11

struct PackStage4 {
    static Stage4State pack() {
        u32 cp = 0;
        for (int i = 0; i < 8; ++i) {
            u32 piece = (g_rawCornerState >> (5*i)) & 7u;
            cp |= piece << (3*i);
        }
        u64 ep = 0;
        for (int i = 0; i < 12; ++i) {
            u64 piece = (g_rawEdgeState >> (5*i)) & 0x1Full;
            ep |= piece << (4*i);
        }
        return { cp, ep };
    }
};

__device__ inline u64 applyEdgePermFull(u64 state, int move) {
    u64 out = 0;
    #pragma unroll
    for (int d = 0; d < 12; ++d) {
        int s = d_edgePermutation[move][d];
        u64 piece = (state >> (4*s)) & 0xFull;
        out |= piece << (4*d);
    }
    return out;
}

struct ApplyStage4 {
    static __device__ void apply(Stage4State &st, int move) {
        st.cornerState = applyCornerPerm(st.cornerState, move);
        st.edgeState   = applyEdgePermFull(st.edgeState, move);
    }
};

struct IsSolved4 {
    static __device__ bool check(const Stage4State &st) {
        return st.cornerState == IDENTITY_CORNER
            && st.edgeState   == IDENTITY_EDGE;
    }
};

std::vector<std::string> solveStage4() {
    const int* d_moves4 = nullptr;
    cudaGetSymbolAddress((void**)&d_moves4, d_allowedMovesStage4);
    return solveStage<6, Stage4State, ApplyStage4, IsSolved4, 15>(
        PackStage4::pack, 15, d_moves4
    );
}
