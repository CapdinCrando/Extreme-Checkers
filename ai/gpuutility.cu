#ifdef CUDA_EDIT
#define CUDA_KERNEL(...)
#define __syncthreads()
#else
#define CUDA_KERNEL(...) <<<__VA_ARGS__>>>
#endif



#include "gpuutility.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>

__constant__ const boardpos_t cornerListDev[SQUARE_COUNT][4] = {
	{-1, -1, 4, 5},{-1, -1, 5, 6},{-1, -1, 6, 7},{-1, -1, 7, -1},
	{-1, 0, -1, 8},{0, 1, 8, 9},{1, 2, 9, 10},{2, 3, 10, 11},
	{4, 5, 12, 13},{5, 6, 13, 14},{6, 7, 14, 15},{7, -1, 15, -1},
	{-1, 8, -1, 16},{8, 9, 16, 17},{9, 10, 17, 18},{10, 11, 18, 19},
	{12, 13, 20, 21},{13, 14, 21, 22},{14, 15, 22, 23},{15, -1, 23, -1},
	{-1, 16, -1, 24},{16, 17, 24, 25},{17, 18, 25, 26},{18, 19, 26, 27},
	{20, 21, 28, 29},{21, 22, 29, 30},{22, 23, 30, 31},{23, -1, 31, -1},
	{-1, 24, -1, -1},{24, 25, -1, -1},{25, 26, -1, -1},{26, 27, -1, -1}
};

__global__ void printKernel()
{
	printf("Hello from mykernel\n");
}

__global__ void getAllBlackMovesKernel(Move* moveList, boardstate_t* board)
{
	__shared__ boardstate_t boardTile[SQUARE_COUNT];
	__shared__ Move moveTile[4*SQUARE_COUNT];
	__shared__ boardpos_t cornerTile[SQUARE_COUNT][4];

	unsigned int i = threadIdx.x;

	boardTile[i] = board[i];
	cornerTile[i][0] = cornerListDev[i][0];
	cornerTile[i][1] = cornerListDev[i][1];
	cornerTile[i][2] = cornerListDev[i][2];
	cornerTile[i][3] = cornerListDev[i][3];
	__syncthreads();

	Move* moveBlock = &moveTile[i*4];
	boardstate_t state = boardTile[i];
	if(SQUARE_ISBLACK(state))
	{
		uint8_t cornerMin = 2;
		if(SQUARE_ISKING(state)) cornerMin = 0;
		for(uint8_t j = cornerMin; j < 4; j++)
		{
			// Get move
			boardpos_t move = cornerTile[i][j];
			// Check if position is invalid
			if(move != BOARD_POS_INVALID)
			{
				// Check if space is empty
				boardstate_t moveState = boardTile[move];
				if(SQUARE_ISEMPTY(moveState))
				{
					// Add move to potential moves
					moveBlock[j].oldPos = i;
					moveBlock[j].newPos = move;
					moveBlock[j].moveType = MOVE_MOVE;
					//moves->push_back(m);
					//cudaMemcpyAsync(&moveBlock[j], &m, sizeof(Move), cudaMemcpyDeviceToDevice);
				}
				else if(!(SQUARE_ISBLACK(moveState)))
				{
					// Get jump
					boardpos_t jump = cornerTile[move][j];
					// Check if position is invalid
					if(jump != BOARD_POS_INVALID)
					{
						// Check if space is empty
						if(SQUARE_ISEMPTY(boardTile[jump]))
						{
							// Add move to potential moves
							moveBlock[j].newPos = jump;
							moveBlock[j].jumpPos = move;
							// Check for multi
							moveBlock[j].moveType = MOVE_JUMP;
							for(uint8_t k = 0; k < 4; k++)
							{
								boardpos_t moveMulti = cornerTile[jump][k];
								// Check if position is invalid
								if(moveMulti != BOARD_POS_INVALID)
								{
									if(moveMulti != move)
									{
										boardstate_t moveStateMulti = boardTile[moveMulti];
										if(SQUARE_ISNOTEMPTY(moveStateMulti))
										{
											if(!(SQUARE_ISBLACK(moveStateMulti)))
											{
												boardpos_t jumpMulti = cornerTile[moveMulti][k];
												if(jumpMulti != BOARD_POS_INVALID)
												{
													boardstate_t jumpStateMulti = boardTile[jumpMulti];
													if(SQUARE_ISEMPTY(jumpStateMulti))
													{
														moveBlock[j].moveType = MOVE_JUMP_MULTI;
														break;
													}
												}
											}
										}
									}
								}
							}
							//cudaMemcpyAsync(&moveBlock[j], &m, sizeof(Move), cudaMemcpyDeviceToDevice);
						}
					}
				}
			}
		}
	}
	unsigned int i4 = i*4;
	moveList[i4] = moveTile[i4];
	moveList[i4+1] = moveTile[i4+1];
	moveList[i4+2] = moveTile[i4+2];
	moveList[i4+3] = moveTile[i4+3];
}

std::vector<Move>* GPUUtility::getAllBlackMoves(BoardState* board)
{
	Move* moves_dev;
	cudaMalloc(&moves_dev, 4*32*sizeof(Move));
	cudaMemset(&moves_dev, 0, 4*32*sizeof(Move));

	boardstate_t* board_dev;
	cudaMalloc(&board_dev, sizeof(BoardState));
	cudaMemcpy(board_dev, board, sizeof(BoardState), cudaMemcpyHostToDevice);

#ifdef PROFILING
	cudaEvent_t start_k, stop_k;

	cudaEventCreate(&start_k);
	cudaEventCreate(&stop_k);
	cudaEventRecord(start_k);
#endif
	getAllBlackMovesKernel CUDA_KERNEL(1,32) (moves_dev, board_dev);
	cudaDeviceSynchronize();
#ifdef PROFILING
	cudaEventRecord(stop_k);
#endif

	Move* moves_host = new Move[32*4];
	cudaMemcpy(moves_host, moves_dev, 32*4*sizeof(Move), cudaMemcpyDeviceToHost);
	cudaFree(moves_dev);
	cudaFree(board_dev);



	std::vector<Move>* moves = new std::vector<Move>;
	std::vector<Move>* jumps = new std::vector<Move>;
	for(uint8_t i = 0; i < 32*4; i++)
	{
		Move m = moves_host[i];
		if(m.moveType != MOVE_INVALID)
		{
			if(MOVE_ISJUMP(m))
			{
				jumps->push_back(m);
			}
			else moves->push_back(m);
		}
	}

#ifdef PROFILING
	cudaEventSynchronize(stop_k);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_k, stop_k);
	printf("Kernel executed in %.3f us\n", milliseconds*1000);
#endif

	delete[] moves_host;
	if(jumps->empty())
	{
		delete jumps;
		return moves;
	}
	delete moves;
	return jumps;
}

void GPUUtility::testPrint()
{
	printKernel CUDA_KERNEL(1,8) ();
	cudaDeviceSynchronize();
}
