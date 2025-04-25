#pragma once
#include <cstdint>
#include <cuda_runtime.h>

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;

// Device constant: corner permutation for each move (dest ← src)
__device__ __constant__ int d_cornerPermutation[18][8] = {
    {3,0,1,2,4,5,6,7}, {2,3,0,1,4,5,6,7}, {1,2,3,0,4,5,6,7}, // U,U2,U'
    {4,1,2,0,7,5,6,3}, {7,1,2,4,3,5,6,0}, {3,1,2,7,0,5,6,4}, // R,R2,R'
    {1,5,2,3,0,4,6,7}, {5,4,2,3,1,0,6,7}, {4,0,2,3,5,1,6,7}, // F,F2,F'
    {0,1,2,3,5,6,7,4}, {0,1,2,3,6,7,4,5}, {0,1,2,3,7,4,5,6}, // D,D2,D'
    {0,2,6,3,4,1,5,7}, {0,6,5,3,4,2,1,7}, {0,5,1,3,4,6,2,7}, // L,L2,L'
    {0,1,3,7,4,5,2,6}, {0,1,7,6,4,5,3,2}, {0,1,6,2,4,5,7,3}  // B,B2,B'
};

// Device constant: how each corner’s orientation changes (after permutation)
__device__ __constant__ int d_cornerOrientation[18][8] = {
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, // U,U2,U'
    {2,0,0,1,1,0,0,2}, {0,0,0,0,0,0,0,0}, {2,0,0,1,1,0,0,2}, // R,R2,R'
    {1,2,0,0,2,1,0,0}, {0,0,0,0,0,0,0,0}, {1,2,0,0,2,1,0,0}, // F,F2,F'
    {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, // D,D2,D'
    {0,1,2,0,0,2,1,0}, {0,0,0,0,0,0,0,0}, {0,1,2,0,0,2,1,0}, // L,L2,L'
    {0,0,1,2,0,0,2,1}, {0,0,0,0,0,0,0,0}, {0,0,1,2,0,0,2,1}  // B,B2,B'
};

// Device constant: edge permutation for each move (dest ← src)
__device__ __constant__ int d_edgePermutation[18][12] = {
    {3,0,1,2,4,5,6,7,8,9,10,11}, {2,3,0,1,4,5,6,7,8,9,10,11}, {1,2,3,0,4,5,6,7,8,9,10,11}, // U,U2,U'
    {8,1,2,3,11,5,6,7,4,9,10,0}, {4,1,2,3,0,5,6,7,11,9,10,8}, {11,1,2,3,8,5,6,7,0,9,10,4}, // R,R2,R'
    {0,9,2,3,4,8,6,7,1,5,10,11}, {0,5,2,3,4,1,6,7,9,8,10,11}, {0,8,2,3,4,9,6,7,5,1,10,11}, // F,F2,F'
    {0,1,2,3,5,6,7,4,8,9,10,11}, {0,1,2,3,6,7,4,5,8,9,10,11}, {0,1,2,3,7,4,5,6,8,9,10,11}, // D,D2,D'
    {0,1,10,3,4,5,9,7,8,2,6,11}, {0,1,6,3,4,5,2,7,8,10,9,11}, {0,1,9,3,4,5,10,7,8,6,2,11}, // L,L2,L'
    {0,1,2,11,4,5,6,10,8,9,3,7}, {0,1,2,7,4,5,6,3,8,9,11,10}, {0,1,2,10,4,5,6,11,8,9,7,3}  // B,B2,B'
};

// Device constant: which edges get flipped by each move
__device__ __constant__ u16 d_edgeOrientation[18] = {
    0b0000'0000'0000, 0b0000'0000'0000, 0b0000'0000'0000, // U,U2,U'
    0b0000'0000'0000, 0b0000'0000'0000, 0b0000'0000'0000, // R,R2,R'
    0b0011'0010'0010, 0b0000'0000'0000, 0b0011'0010'0010, // F,F2,F'
    0b0000'0000'0000, 0b0000'0000'0000, 0b0000'0000'0000, // D,D2,D'
    0b0000'0000'0000, 0b0000'0000'0000, 0b0000'0000'0000, // L,L2,L'
    0b1100'1000'1000, 0b0000'0000'0000, 0b1100'1000'1000  // B,B2,B'
};


// Slice‐mask helper
__device__ __constant__ u16 d_equatorSliceMask = 0b1111'0000'0000;
__device__ __constant__ u16 d_middleSliceMask = 0b0000'0101'0101;

// Human‐readable move names
static constexpr const char* move_names[18] = {
    "U","U2","U'", "R","R2","R'", "F","F2","F'",
    "D","D2","D'", "L","L2","L'", "B","B2","B'"
};

//— Generic device‐only helpers —//

__device__ inline u32 applyCornerPerm(u32 state, int mv) {
    u32 out = 0;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        int s = d_cornerPermutation[mv][d];
        u32 piece = (state >> (3 * s)) & 7u;
        out |= piece << (3 * d);
    }
    return out;
}

__device__ inline u16 applyEdgePerm(u16 state, int mv) {
    u16 out = 0;
    #pragma unroll
    for (int d = 0; d < 12; ++d) {
        int s = d_edgePermutation[mv][d];
        u16 bit = (state >> s) & 1u;
        out |= bit << d;
    }
    return out;
}

__device__ inline u32 applyCornerTwist(u32 state, int mv) {
    u32 out = 0;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        // unpack 5 bits: low 3 = piece, high 2 = orientation
        u32 packed    = (state >> (5 * d)) & 0x1Fu;
        u32 pieceIdx  = packed & 0x7u;
        u32 ori       = (packed >> 3) & 0x3u;
        u32 newOri    = (ori + d_cornerOrientation[mv][d]) % 3u;
        u32 outPacked = pieceIdx | (newOri << 3);
        out |= outPacked << (5 * d);
    }
    return out;
}

__device__ u16 applyCornerTwistMoveGpu(u16 currentCornerState, int moveIndex) {
    u16 newCornerState = 0;
    for (int dst = 0; dst < 8; ++dst) {
        int src     = d_cornerPermutation[moveIndex][dst];
        int ori     = (currentCornerState >> (2*src)) & 0x3;
        int twist   = d_cornerOrientation[moveIndex][dst];
        int newOri  = (ori + twist) % 3;
        newCornerState |= u16(newOri) << (2*dst);
    }
    return newCornerState;
}

__device__ inline u16 applyEdgeFlip(u16 state, int mv) {
    u16 out = 0;
    #pragma unroll
    for (int d = 0; d < 12; ++d) {
        int s = d_edgePermutation[mv][d];
        u16 ori = (state >> s) & 1u;
        ori ^= (d_edgeOrientation[mv] >> d) & 1u;
        out |= ori << d;
    }
    return out;
}
