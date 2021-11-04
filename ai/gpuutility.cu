#ifdef CUDA_EDIT
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<<__VA_ARGS__>>>
#endif

#define GET_BOARD_STATE_KERNEL(board, pos) (board[pos/2] >> (pos%2)*4) & 0x7

#include "gpuutility.h"

#include <cuda_runtime.h>
#include <device_functions.h>
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
	unsigned int i = threadIdx.x;

	Move* moveListBlock = &moveList[i*4];

	Move m;
	m.oldPos = i;
	boardstate_t state = GET_BOARD_STATE_KERNEL(board, i);
	if(SQUARE_ISBLACK(state))
	{
		uint8_t cornerMin = 2;
		if(SQUARE_ISKING(state)) cornerMin = 0;
		for(uint8_t j = cornerMin; j < 4; j++)
		{
			// Get move
			boardpos_t move = cornerListDev[i][j];
			// Check if position is invalid
			if(move != BOARD_POS_INVALID)
			{
				// Check if space is empty
				boardstate_t moveState = GET_BOARD_STATE_KERNEL(board, move);
				if(SQUARE_ISEMPTY(moveState))
				{
					// Add move to potential moves
					m.newPos = move;
					m.moveType = MOVE_MOVE;
					//moves->push_back(m);
					moveListBlock[j] = m;
				}
				else if(!(SQUARE_ISBLACK(moveState)))
				{
					// Get jump
					boardpos_t jump = cornerListDev[move][j];
					// Check if position is invalid
					if(jump != BOARD_POS_INVALID)
					{
						// Check if space is empty
						if(SQUARE_ISEMPTY(GET_BOARD_STATE_KERNEL(board, jump)))
						{
							// Add move to potential moves
							m.newPos = jump;
							m.jumpPos = move;
							// Check for multi
							m.moveType = MOVE_JUMP;
							for(uint8_t k = 0; k < 4; k++)
							{
								boardpos_t moveMulti = cornerListDev[jump][k];
								// Check if position is invalid
								if(moveMulti != BOARD_POS_INVALID)
								{
									if(moveMulti != move)
									{
										boardstate_t moveStateMulti = GET_BOARD_STATE_KERNEL(board, moveMulti);
										if(SQUARE_ISNOTEMPTY(moveStateMulti))
										{
											if(!(SQUARE_ISBLACK(moveStateMulti)))
											{
												boardpos_t jumpMulti = cornerListDev[moveMulti][k];
												if(jumpMulti != BOARD_POS_INVALID)
												{
													boardstate_t jumpStateMulti = GET_BOARD_STATE_KERNEL(board, jumpMulti);
													if(SQUARE_ISEMPTY(jumpStateMulti))
													{
														m.moveType = MOVE_JUMP_MULTI;
														break;
													}
												}
											}
										}
									}
								}
							}
							moveListBlock[j] = m;
						}
					}
				}
			}
		}
	}
}

std::vector<Move>* GPUUtility::getAllBlackMoves(BoardState* board)
{
	Move* moves_dev;
	cudaMalloc(&moves_dev, 4*12*sizeof(Move));
	cudaMemset(&moves_dev, 0, 4*12*sizeof(Move));

	boardstate_t* board_dev;
	cudaMalloc(&board_dev, sizeof(BoardState));
	cudaMemcpy(board_dev, board, sizeof(BoardState), cudaMemcpyHostToDevice);

	getAllBlackMovesKernel CUDA_KERNEL(1,32) (moves_dev, board_dev);

	Move* moves_host = new Move[12*4];
	cudaMemcpy(moves_host, moves_dev, 12*4*sizeof(Move), cudaMemcpyDeviceToHost);

	std::vector<Move>* moves = new std::vector<Move>;
	for(uint8_t i = 0; i < 12*4; i++)
	{
		Move m = moves_host[i];
		if(m.moveType != MOVE_INVALID)
		{
			moves->push_back(m);
		}
	}
	return moves;
}
void GPUUtility::testPrint()
{
	printKernel CUDA_KERNEL(1,8) ();
	cudaDeviceSynchronize();
}
