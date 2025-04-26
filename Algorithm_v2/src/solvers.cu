#include "solver_kernel.cuh"
#include "cube_defs.cuh"
#include "common.h"
#include <vector>
#include <string>
#include <cstdint>

__device__ __constant__ int d_allowedMovesStage1[18] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
__device__ __constant__ int d_allowedMovesStage2[14] = {0,1,2,3,4,5,7,9,10,11,12,13,14,16};
__device__ __constant__ int d_allowedMovesStage3[10] = {0,1,2,4,7,9,10,11,13,16};
__device__ __constant__ int d_allowedMovesStage4[6] = {1,4,7,10,13,16};

//=== Stage 1: Edge Orientation ===

struct ApplyEdgeOrientation {
    __device__ static void apply(u16 &state, int move) {
        state = applyEdgeOrientation(state, move);
    }
};

struct EdgeOrientationSolved {
    __device__ static bool check(u16 state) {
        return state == 0;
    }
};

u16 packEdgeOrientationState() {
    u16 packed = 0;
    for (int i = 0; i < 12; ++i) {
        u16 bit = u16((g_rawEdgeState >> (5 * i + 4) & 1u));
        packed |= bit << i;
    }
    return packed;
}

std::vector<std::string> solveStage1() {
    const int *d_moves = nullptr;
    cudaGetSymbolAddress((void **)&d_moves, d_allowedMovesStage1);
    return solveStage<18, u16, ApplyEdgeOrientation, EdgeOrientationSolved,7>(
        packEdgeOrientationState, 7, d_moves
    );
}

//=== Stage 2: Domino Reduction ===

struct PackCornerTwistAndEquatorSliceState {
    static u32 pack() {
        u16 twist = 0;
        for (int i = 0; i < 8; ++i) {
            u32 slot = (g_rawCornerState >> (5 * i)) & 0x1Fu;
            u16 ori = u16((slot >> 3) & 0x3u);
            twist |= ori << (2 * i);
        }
        u16 slice = 0;
        for (int i = 0; i < 12; ++i) {
            u32 slot = (g_rawEdgeState >> (5 * i)) & 0x1Fu;
            bool inEquator = (slot >= 8 && slot <= 11);
            slice |= u16(inEquator) << i;
        }
        return u32(twist) | (u32(slice) << 16);
    }
};

struct ApplyCornerTwistAndEquatorSlice {
    __device__ static void apply(u32 &state, int move) {
        u16 orientation = u16(state & 0xFFFFu);
        u16 permutation = u16(state >> 16);
        orientation = applyCornerOrientation(orientation, move);
        permutation = applyEdgePerm(permutation, move);
        state = u32(orientation) | (u32(permutation) << 16);
    }
};

struct CornerTwistAndEquatorSliceSolved {
    __device__ static bool check(u32 state) {
        u16 orientation = u16(state & 0xFFFFu);
        u16 permutation = u16(state >> 16);
        return orientation == 0 && (permutation & d_equatorSliceMask) == d_equatorSliceMask;
    }
};

std::vector<std::string> solveStage2() {
    const int *d_moves = nullptr;
    cudaGetSymbolAddress((void **)&d_moves, d_allowedMovesStage2);
    return solveStage<14, u32, ApplyCornerTwistAndEquatorSlice, CornerTwistAndEquatorSliceSolved, 10>(
        PackCornerTwistAndEquatorSliceState::pack, 10, d_moves
    );
}

//=== Stage 3: Half Turn Reduction ===

struct PackCornerPermutationAndMiddleSliceState {
    static u64 pack() {
        u32 cornerPack = 0;
        for (int i = 0; i < 8; ++i) {
            u32 piece = (g_rawCornerState >> (5 * i)) & 7u;
            cornerPack |= piece << (3 * i);
        }
        u16 sliceMask = 0;
        for (int i = 0; i < 12; ++i) {
            u32 packed = (g_rawEdgeState >> (5 * i)) & 0x1Fu;
            u32 piece = packed & 0xFu;
            bool inMiddle = (piece == 0 || piece == 2 || piece == 4 || piece == 6);
            sliceMask |= u16(inMiddle) << i;
        }
        return u64(cornerPack) | (u64(sliceMask) << 32);
    }
};

struct ApplyCornerPermutationAndMiddleSlice {
    __device__ static void apply(u64 &state, int move) {
        u32 corners = u32(state);
        u16 slice = u16(state >> 32);
        corners = applyCornerPerm(corners, move);
        slice = applyEdgePerm(slice, move);
        state = u64(corners) | (u64(slice) << 32);
    }
};

struct CornerPermutationAndMiddleSliceSolved {
    __device__ static bool check(u64 state) {
        u32 corners = u32(state);
        u16 slice = u16(state >> 32);
        if ((slice & d_middleSliceMask) != d_middleSliceMask) {
            return false;
        }
        int array[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            array[i] = (corners >> (3 * i)) & 7;
        }
        int parity = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            for (int j = i + 1; j < 8; ++j) {
                parity ^= (array[i] > array[j]);
            }
        }
        if (parity != 0) {
            return false;
        }
        unsigned maskAccum = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int piece = (corners >> (3 * i)) & 7;
            maskAccum |= ((piece ^ i) & 5u);
        }
        return maskAccum == 0;
    }
};

std::vector<std::string> solveStage3() {
    const int *d_moves = nullptr;
    cudaGetSymbolAddress((void **)&d_moves, d_allowedMovesStage3);
    return solveStage<10, u64, ApplyCornerPermutationAndMiddleSlice, CornerPermutationAndMiddleSliceSolved, 13>(
        PackCornerPermutationAndMiddleSliceState::pack, 13, d_moves
    );
}

//=== Stage 4: Solved ===

struct FinalPermutationState {
    u32 cornerState;
    u64 edgeState;
};

static constexpr u32 IdentityCorner = 0xFAC688u;
static constexpr u64 IdentityEdge   = 0xBA9876543210ull;

struct PackFinalPermutationState {
    static FinalPermutationState pack() {
        u32 cp = 0;
        for (int i = 0; i < 8; ++i) {
            u32 piece = (g_rawCornerState >> (5 * i)) & 7u;
            cp |= piece << (3 * i);
        }
        u64 ep = 0;
        for (int i = 0; i < 12; ++i) {
            u64 piece = (g_rawEdgeState >> (5 * i)) & 0x1Full;
            ep |= piece << (4 * i);
        }
        return {cp, ep};
    }
};

struct ApplyFinalPermutation {
    __device__ static void apply(FinalPermutationState &st, int move) {
        st.cornerState = applyCornerPerm(st.cornerState, move);
        st.edgeState = applyFullEdgePermutation(st.edgeState, move);
    }
};

struct FinalPermutationSolved {
    __device__ static bool check(const FinalPermutationState &st) {
        return st.cornerState == IdentityCorner && st.edgeState   == IdentityEdge;
    }
};

std::vector<std::string> solveStage4() {
    const int *d_moves = nullptr;
    cudaGetSymbolAddress((void **)&d_moves, d_allowedMovesStage4);
    return solveStage<6, FinalPermutationState, ApplyFinalPermutation, FinalPermutationSolved, 15>(
        PackFinalPermutationState::pack, 15, d_moves
    );
}
