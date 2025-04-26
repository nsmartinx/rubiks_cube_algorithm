#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "cube_defs.cuh"

// Kernel: brute-force search
template<int MoveCount,
         typename StateType,
         typename ApplyMove,
         typename IsSolvedPred,
         int MaxDepth>
__global__
void bruteForceKernel(
    StateType    start_state,
    const int*   allowed_moves,
    int          depth,
    int*         solution_buffer,
    int*         found_flag
) {
    // compute flat thread index
    unsigned long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // total sequences = MoveCount^depth
    unsigned long long total_sequences = 1;
    for (int i = 0; i < depth; ++i) {
        total_sequences *= MoveCount;
    }
    if (thread_id >= total_sequences) {
        return;
    }

    StateType state = start_state;
    int sequence[MaxDepth];
    int previous_face = -1;
    unsigned long long sequence_code = thread_id;

    // decode moves and apply
    for (int step = 0; step < depth; ++step) {
        int selection = sequence_code % MoveCount;
        sequence_code /= MoveCount;
        int move = allowed_moves[selection];
        int face = move / 3;
        if (face == previous_face) {
            return;
        }
        previous_face = face;
        sequence[step] = move;
        ApplyMove::apply(state, move);
    }

    // check solved and record
    if (IsSolvedPred::check(state)) {
        if (atomicExch(found_flag, 1) == 0) {
            for (int i = 0; i < depth; ++i) {
                solution_buffer[i] = sequence[i];
            }
        }
    }
}

// Host driver: iterative deepening
template<int MoveCount,
         typename StateType,
         typename ApplyMove,
         typename IsSolvedPred,
         int MaxDepth>
std::vector<std::string> solveStage(
    StateType    (*pack_state)(),
    int           max_depth,
    const int*    d_allowed_moves
) {
    StateType start_state = pack_state();

    int* d_solution_buffer = nullptr;
    int* d_found_flag = nullptr;
    cudaMalloc(&d_solution_buffer, sizeof(int) * MaxDepth);
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMemset(d_found_flag, 0, sizeof(int));

    int h_sequence[MaxDepth] = {0};
    int h_found_flag = 0;

    for (int depth = 1; depth <= max_depth; ++depth) {
        // compute number of blocks needed
        unsigned long long total_sequences = 1;
        for (int i = 0; i < depth; ++i) {
            total_sequences *= MoveCount;
        }
        int block_count = int((total_sequences + 255) / 256);

        // launch kernel
        bruteForceKernel<MoveCount, StateType, ApplyMove, IsSolvedPred, MaxDepth>
            <<<block_count, 256>>>(
                start_state,
                d_allowed_moves,
                depth,
                d_solution_buffer,
                d_found_flag
            );
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "[solveStage] kernel launch failed: "
                      << cudaGetErrorString(error) << std::endl;
            break;
        }

        cudaDeviceSynchronize();

        // check for solution
        cudaMemcpy(&h_found_flag, d_found_flag, sizeof(h_found_flag), cudaMemcpyDeviceToHost);
        if (h_found_flag) {
            cudaMemcpy(h_sequence, d_solution_buffer, sizeof(int) * depth, cudaMemcpyDeviceToHost);
            std::vector<std::string> solution;
            solution.reserve(depth);
            for (int i = 0; i < depth; ++i) {
                solution.push_back(move_names[h_sequence[i]]);
            }
            cudaFree(d_solution_buffer);
            cudaFree(d_found_flag);
            return solution;
        }
    }

    cudaFree(d_solution_buffer);
    cudaFree(d_found_flag);
    std::cerr << "ERROR: No solution found" << std::endl;
    return {};
}
