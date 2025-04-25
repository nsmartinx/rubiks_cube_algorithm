#pragma once
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "cube_defs.cuh"
#include <iostream>

template<int MoveCount,
         typename StateType,
         typename ApplyMove,     // functor: ApplyMove::apply(state,move)
         typename IsSolvedPred,  // functor: IsSolvedPred::check(state)
         int MaxDepth>
__global__
void bruteForceKernel(
    StateType    startState,
    const int   *allowedMoves,
    int          depth,
    int         *solutionBuf,
    int         *foundFlag
) {
    unsigned long long idx   = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned long long total = 1;
    for(int i=0;i<depth;++i) total *= MoveCount;
    if (idx >= total) return;

    StateType st = startState;
    int seq[MaxDepth];
    int prevFace = -1;
    unsigned long long code = idx;

    for(int d=0; d<depth; ++d){
        int sel = code % MoveCount;
        code  /= MoveCount;
        int mv  = allowedMoves[sel];
        int face = mv/3;
        if(face == prevFace) return;
        prevFace = face;
        seq[d] = mv;
        ApplyMove::apply(st, mv);
    }

    if(IsSolvedPred::check(st)){
        if(atomicExch(foundFlag,1) == 0){
            for(int i=0;i<depth;++i)
                solutionBuf[i] = seq[i];
        }
    }
}

template<int MoveCount,
         typename StateType,
         typename ApplyMove,
         typename IsSolvedPred,
         int MaxDepth>
std::vector<std::string> solveStage(
    StateType    (*packState)(),
    int           maxDepth,
    const int    *d_allowedMoves
) {
    StateType start = packState();

    int *d_buf, *d_flag;
    cudaMalloc(&d_buf,  sizeof(int)*MaxDepth);
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag,0,sizeof(int));

    int hostSeq[MaxDepth] = {0}, found=0;
    for(int depth=1; depth<=maxDepth; ++depth){
        unsigned long long total = 1;
        for(int i=0;i<depth;++i) total *= MoveCount;

        int blocks = int((total + 255)/256);
        bruteForceKernel<MoveCount,StateType,ApplyMove,IsSolvedPred,MaxDepth>
          <<<blocks,256>>>(start,
                           d_allowedMoves,
                           depth,
                           d_buf,
                           d_flag);
                           cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          std::cerr << "[solveStage] kernel launch failed: "
                    << cudaGetErrorString(err) << "\n";
            break;
        }

        cudaDeviceSynchronize();

        cudaMemcpy(&found, d_flag, sizeof(found), cudaMemcpyDeviceToHost);
        if(found){
            cudaMemcpy(hostSeq, d_buf, sizeof(int)*depth, cudaMemcpyDeviceToHost);
            std::vector<std::string> sol;
            sol.reserve(depth);
            for(int i=0;i<depth;++i)
                sol.push_back(move_names[hostSeq[i]]);
            cudaFree(d_buf);
            cudaFree(d_flag);
            return sol;
        }
    }

    cudaFree(d_buf);
    cudaFree(d_flag);
    std::cout << "ERROR: No solution found" << std::endl;
    return {};
}
