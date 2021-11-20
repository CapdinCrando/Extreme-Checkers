#include "gpuutility.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "aiutility.h"

// To get rid of intellisense warnings

#ifdef CUDA_EDIT
#define CUDA_KERNEL(...)
#define __syncthreads()
template<class T1, class T2>
__device__ void atomicMax(T1 x, T2 y);
template<class T1, class T2>
__device__ void atomicMin(T1 x, T2 y);
template<class T, class T2>
__device__ T atomicAdd(T* x, T2 y);
template<class T, class T2>
__device__ T atomicCAS(T* x, T2 y, T2 z);
#else
#define CUDA_KERNEL(...) <<<__VA_ARGS__>>>
#endif

#ifdef QT_DEBUG
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__device__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
	  printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}
#else
#define gpuErrchk(ans) (ans)
#endif

#define IS_ROOT_THREAD threadIdx.x == 0U
#define MOVE_BUFFER_SIZE SQUARE_COUNT

__device__ boardpos_t previousMultiJumpPosGPU = -1;

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


__device__ void getBlackJumpsGPU(Move* jumpsOut, unsigned int& jumpCount, boardpos_t pos, boardstate_t* board, boardpos_t (&cornerTile)[SQUARE_COUNT][4])
{
	__shared__ Move jumps[MOVE_BUFFER_SIZE];
	unsigned int i = pos;

	unsigned int j = threadIdx.x;
	if(j < 4)
	{
		boardstate_t state = board[i];
		if(SQUARE_ISNOTEMPTY(state))
		{
			if(SQUARE_ISBLACK(state))
			{
				// Get move
				boardpos_t move = cornerTile[i][j];
				// Check if position is invalid
				if(move != BOARD_POS_INVALID)
				{
					// Check if space is empty
					boardstate_t moveState = board[move];
					if(SQUARE_ISNOTEMPTY(moveState))
					{
						if(!(SQUARE_ISBLACK(moveState)))
						{
							// Get jump
							boardpos_t jump = cornerTile[move][j];
							// Check if position is invalid
							if(jump != BOARD_POS_INVALID)
							{
								// Check if space is empty
								if(SQUARE_ISEMPTY(board[jump]))
								{
									// Add move to potential moves
									uint16_t jumpIndex = atomicAdd(&jumpCount, 1U);
									jumps[jumpIndex].oldPos = i;
									jumps[jumpIndex].newPos = jump;
									jumps[jumpIndex].jumpPos = move;
									// Check for multi
									jumps[jumpIndex].moveType = MOVE_JUMP;
									for(uint8_t k = 0; k < 4; k++)
									{
										boardpos_t moveMulti = cornerTile[jump][k];
										// Check if position is invalid
										if(moveMulti != BOARD_POS_INVALID)
										{
											if(moveMulti != move)
											{
												boardstate_t moveStateMulti = board[moveMulti];
												if(SQUARE_ISNOTEMPTY(moveStateMulti))
												{
													if(!(SQUARE_ISBLACK(moveStateMulti)))
													{
														boardpos_t jumpMulti = cornerTile[moveMulti][k];
														if(jumpMulti != BOARD_POS_INVALID)
														{
															boardstate_t jumpStateMulti = board[jumpMulti];
															if(SQUARE_ISEMPTY(jumpStateMulti))
															{
																jumps[jumpIndex].moveType = MOVE_JUMP_MULTI;
																break;
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	__syncthreads();
	if(j < jumpCount) jumpsOut[j] = jumps[j];
	__syncthreads();
}

__device__ void getRedJumpsGPU(Move* jumpsOut, unsigned int& jumpCount, boardpos_t pos, boardstate_t* board, boardpos_t (&cornerTile)[SQUARE_COUNT][4])
{
	__shared__ Move jumps[MOVE_BUFFER_SIZE];
	unsigned int i = pos;

	unsigned int j = threadIdx.x;
	if(j < 4)
	{
		boardstate_t state = board[i];
		if(SQUARE_ISNOTEMPTY(state))
		{
			if(!(SQUARE_ISBLACK(state)))
			{
				// Get move
				boardpos_t move = cornerTile[i][j];
				// Check if position is invalid
				if(move != BOARD_POS_INVALID)
				{
					// Check if space is empty
					boardstate_t moveState = board[move];
					if(SQUARE_ISNOTEMPTY(moveState))
					{
						if(SQUARE_ISBLACK(moveState))
						{
							// Get jump
							boardpos_t jump = cornerTile[move][j];
							// Check if position is invalid
							if(jump != BOARD_POS_INVALID)
							{
								// Check if space is empty
								if(SQUARE_ISEMPTY(board[jump]))
								{
									// Add move to potential moves
									uint16_t jumpIndex = atomicAdd(&jumpCount, 1U);
									jumps[jumpIndex].oldPos = i;
									jumps[jumpIndex].newPos = jump;
									jumps[jumpIndex].jumpPos = move;
									// Check for multi
									jumps[jumpIndex].moveType = MOVE_JUMP;
									for(uint8_t k = 0; k < 4; k++)
									{
										boardpos_t moveMulti = cornerTile[jump][k];
										// Check if position is invalid
										if(moveMulti != BOARD_POS_INVALID)
										{
											if(moveMulti != move)
											{
												boardstate_t moveStateMulti = board[moveMulti];
												if(SQUARE_ISNOTEMPTY(moveStateMulti))
												{
													if(SQUARE_ISBLACK(moveStateMulti))
													{
														boardpos_t jumpMulti = cornerTile[moveMulti][k];
														if(jumpMulti != BOARD_POS_INVALID)
														{
															boardstate_t jumpStateMulti = board[jumpMulti];
															if(SQUARE_ISEMPTY(jumpStateMulti))
															{
																jumps[jumpIndex].moveType = MOVE_JUMP_MULTI;
																break;
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	__syncthreads();
	if(j < jumpCount) jumpsOut[j] = jumps[j];
	__syncthreads();
}

__device__ void getBlackMovesGPU(Move* movesOut, unsigned int& moveCount, boardstate_t* board, boardpos_t (&cornerTile)[SQUARE_COUNT][4])
{
	__shared__ Move moves[MOVE_BUFFER_SIZE];
	__shared__ Move jumps[MOVE_BUFFER_SIZE];
	__shared__ unsigned int jumpCount;
	if(IS_ROOT_THREAD) jumpCount = 0;
	__syncthreads();

	unsigned int i = threadIdx.x;

	boardstate_t state = board[i];
	if(SQUARE_ISNOTEMPTY(state))
	{
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
					boardstate_t moveState = board[move];
					if(SQUARE_ISEMPTY(moveState))
					{
						// Add move to potential moves
						uint16_t moveIndex = atomicAdd(&moveCount, 1U);
						moves[moveIndex].oldPos = i;
						moves[moveIndex].newPos = move;
						moves[moveIndex].moveType = MOVE_MOVE;
					}
					else if(!(SQUARE_ISBLACK(moveState)))
					{
						// Get jump
						boardpos_t jump = cornerTile[move][j];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(SQUARE_ISEMPTY(board[jump]))
							{
								// Add move to potential moves
								uint16_t jumpIndex = atomicAdd(&jumpCount, 1U);
								jumps[jumpIndex].oldPos = i;
								jumps[jumpIndex].newPos = jump;
								jumps[jumpIndex].jumpPos = move;
								// Check for multi
								jumps[jumpIndex].moveType = MOVE_JUMP;
								for(uint8_t k = 0; k < 4; k++)
								{
									boardpos_t moveMulti = cornerTile[jump][k];
									// Check if position is invalid
									if(moveMulti != BOARD_POS_INVALID)
									{
										if(moveMulti != move)
										{
											boardstate_t moveStateMulti = board[moveMulti];
											if(SQUARE_ISNOTEMPTY(moveStateMulti))
											{
												if(!(SQUARE_ISBLACK(moveStateMulti)))
												{
													boardpos_t jumpMulti = cornerTile[moveMulti][k];
													if(jumpMulti != BOARD_POS_INVALID)
													{
														boardstate_t jumpStateMulti = board[jumpMulti];
														if(SQUARE_ISEMPTY(jumpStateMulti))
														{
															jumps[jumpIndex].moveType = MOVE_JUMP_MULTI;
															break;
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	__syncthreads();

	if(jumpCount)
	{
		if(IS_ROOT_THREAD) moveCount = jumpCount;
		if(i < jumpCount) movesOut[i] = jumps[i];
	}
	else if(i < moveCount) movesOut[i] = moves[i];
	__syncthreads();
}

__device__ void getRedMovesGPU(Move* movesOut, unsigned int& moveCount, boardstate_t* board, boardpos_t (&cornerTile)[SQUARE_COUNT][4])
{
	__shared__ Move moves[MOVE_BUFFER_SIZE];
	__shared__ Move jumps[MOVE_BUFFER_SIZE];
	__shared__ unsigned int jumpCount;
	if(IS_ROOT_THREAD) jumpCount = 0;
	__syncthreads();

	unsigned int i = threadIdx.x;

	boardstate_t state = board[i];
	if(SQUARE_ISNOTEMPTY(state))
	{
		if(!(SQUARE_ISBLACK(state)))
		{
			uint8_t cornerMax = 2;
			if(SQUARE_ISKING(state)) cornerMax = 4;
			for(uint8_t j = 0; j < cornerMax; j++)
			{
				// Get move
				boardpos_t move = cornerTile[i][j];
				// Check if position is invalid
				if(move != BOARD_POS_INVALID)
				{
					// Check if space is empty
					boardstate_t moveState = board[move];
					if(SQUARE_ISEMPTY(moveState))
					{
						// Add move to potential moves
						uint16_t moveIndex = atomicAdd(&moveCount, 1U);
						moves[moveIndex].oldPos = i;
						moves[moveIndex].newPos = move;
						moves[moveIndex].moveType = MOVE_MOVE;
					}
					else if(SQUARE_ISBLACK(moveState))
					{
						// Get jump
						boardpos_t jump = cornerTile[move][j];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(SQUARE_ISEMPTY(board[jump]))
							{
								// Add move to potential moves
								uint16_t jumpIndex = atomicAdd(&jumpCount, 1U);
								jumps[jumpIndex].oldPos = i;
								jumps[jumpIndex].newPos = jump;
								jumps[jumpIndex].jumpPos = move;
								// Check for multi
								jumps[jumpIndex].moveType = MOVE_JUMP;
								for(uint8_t k = 0; k < 4; k++)
								{
									boardpos_t moveMulti = cornerTile[jump][k];
									// Check if position is invalid
									if(moveMulti != BOARD_POS_INVALID)
									{
										if(moveMulti != move)
										{
											boardstate_t moveStateMulti = board[moveMulti];
											if(SQUARE_ISNOTEMPTY(moveStateMulti))
											{
												if(SQUARE_ISBLACK(moveStateMulti))
												{
													boardpos_t jumpMulti = cornerTile[moveMulti][k];
													if(jumpMulti != BOARD_POS_INVALID)
													{
														boardstate_t jumpStateMulti = board[jumpMulti];
														if(SQUARE_ISEMPTY(jumpStateMulti))
														{
															jumps[jumpIndex].moveType = MOVE_JUMP_MULTI;
															break;
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	__syncthreads();

	if(jumpCount)
	{
		if(IS_ROOT_THREAD) moveCount = jumpCount;
		if(i < jumpCount) movesOut[i] = jumps[i];
	}
	else if(i < moveCount) movesOut[i] = moves[i];
	__syncthreads();
}

__device__ bool evalBoardSquareGPU(result_gpu_t* resultOut, boardstate_t* board, boardpos_t (&cornerTile)[SQUARE_COUNT][4])
{
	__shared__ int blackCount;
	__shared__ int redCount;
	__shared__ int redMoveFound;
	__shared__ int blackMoveFound;
	__shared__ bool retVal;
	if(IS_ROOT_THREAD)
	{
		blackCount = 0;
		redCount = 0;
		redMoveFound = 0;
		blackMoveFound = 0;
		retVal = false;
	}
	__syncthreads();

	boardpos_t pos = threadIdx.x;
	boardstate_t state = board[pos];
	if(SQUARE_ISNOTEMPTY(state))
	{
		if(SQUARE_ISBLACK(state))
		{
			atomicAdd(&blackCount, 1);
			uint8_t cornerMin = 2;
			if(SQUARE_ISKING(state)) cornerMin = 0;
			for(uint8_t i = cornerMin; i < 4; i++)
			{
				// Get move
				boardpos_t move = cornerTile[pos][i];

				// Check if position is invalid
				if(move != BOARD_POS_INVALID)
				{
					// Check if space is empty
					boardstate_t moveState = board[move];
					if(SQUARE_ISEMPTY(moveState))
					{
						atomicAdd(&blackMoveFound, 1);
					}
					else if(!(SQUARE_ISBLACK(moveState)))
					{
						// Get jump
						boardpos_t jump = cornerTile[move][i];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(SQUARE_ISEMPTY(board[jump]))
							{
								// Add jump to potential moves
								atomicAdd(&blackMoveFound, 1);
							}
						}
					}
				}
			}
		}
		else
		{
			atomicAdd(&redCount, 1);
			uint8_t cornerMax = 2;
			if(SQUARE_ISKING(state)) cornerMax = 4;
			for(uint8_t i = 0; i < cornerMax; i++)
			{
				// Get move
				boardpos_t move = cornerTile[pos][i];

				// Check if position is invalid
				if(move != BOARD_POS_INVALID)
				{
					// Check if space is empty
					boardstate_t moveState = board[move];
					if(SQUARE_ISEMPTY(moveState))
					{
						atomicAdd(&redMoveFound, 1);
					}
					else if(SQUARE_ISBLACK(moveState))
					{
						// Get jump
						boardpos_t jump = cornerTile[move][i];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(SQUARE_ISEMPTY(board[jump]))
							{
								// Add jump to potential moves
								atomicAdd(&redMoveFound, 1);
							}
						}
					}
				}
			}
		}
	}
	__syncthreads();
	printf("%i\n", pos);

	if(IS_ROOT_THREAD)
	{
		if(blackCount == 0)
		{
			if(redCount != 0)
			{
				// Red win
				*resultOut = RESULT_RED_WIN;
				retVal = true;
			}
		}
		else if(redCount == 0)
		{
			if(blackCount != 0)
			{
				// Black win
				*resultOut = RESULT_BLACK_WIN;
				retVal = true;
			}
		}
		else
		{
			if(!blackMoveFound)
			{
				if(redMoveFound)
				{
					// RED WIN
					*resultOut = RESULT_RED_WIN;
					retVal = true;
				}
				else
				{
					// TIE
					*resultOut = RESULT_TIE;
					retVal = true;
				}
			}
			else if(!redMoveFound)
			{
				if(blackMoveFound)
				{
					// BLACK WIN
					*resultOut = RESULT_BLACK_WIN;
					retVal = true;
				}
				else
				{
					// TIE
					*resultOut = RESULT_TIE;
					retVal = true;
				}
			}
		}
		if(!retVal) *resultOut = blackCount - redCount;
	}
	__syncthreads();
	return retVal;
}

// Definition
__device__ void evalRedMoveKernel(result_gpu_t* resultOut, boardstate_t* board, Move* oldMoves, depth_t depth, boardpos_t (&cornerTile)[SQUARE_COUNT][4]);

__device__ void evalBlackMoveKernel(result_gpu_t* resultOut, boardstate_t* board, Move* oldMoves, depth_t depth, boardpos_t (&cornerTile)[SQUARE_COUNT][4])
{
	__shared__ boardstate_t boardTile[SQUARE_COUNT];
	__shared__ Move moveTile;
	__shared__ unsigned int moveCount;
	__shared__ int resultVal;
	__shared__ int resultIndex;
	unsigned int x = threadIdx.x;

	// Copy board
	boardTile[x] = board[x];
	moveTile = *oldMoves;
	boardTile[moveTile.newPos] = boardTile[moveTile.oldPos];
	__syncthreads();

	// Check for king
	boardTile[moveTile.oldPos] = SQUARE_EMPTY;
	if(moveTile.newPos > 27)
	{
		if(SQUARE_ISNOTEMPTY(boardTile[moveTile.newPos]))
		{
			boardTile[moveTile.newPos] |= 0x1;
		}
	}
	if(MOVE_ISJUMP(moveTile)) boardTile[moveTile.jumpPos] = SQUARE_EMPTY;
	moveCount = 0;
	resultIndex = -1;
	__syncthreads();


	// Check depth
	if(evalBoardSquareGPU(resultOut, boardTile, cornerTile)) return;
	if(depth == NODE_DEPTH_GPU)
	{
		return;
	}

	__shared__ Move* moves;
	__shared__ result_gpu_t* results;
	//__shared__ boardstate_t* newBoard;
	if(IS_ROOT_THREAD)
	{
		gpuErrchk(cudaMalloc(&moves, MOVE_BUFFER_SIZE*sizeof(Move)));
		//gpuErrchk(cudaMalloc(&newBoard, SQUARE_COUNT*sizeof(boardstate_t)));
	}
	//__syncthreads();
	//boardTile[x];
	//printf("AHHHH\n");
	//newBoard[x] = boardTile[x];
	//printf("AHHHH\n");
	//__syncthreads();

	if(moveTile.moveType == MOVE_JUMP_MULTI)
	{
		// Create moves
		getBlackJumpsGPU(moves, moveCount, moveTile.newPos, boardTile, cornerTile);

		// Evaluate Moves (recursive)
		if(IS_ROOT_THREAD)
		{
			gpuErrchk(cudaMalloc(&results, moveCount*sizeof(result_gpu_t)));
		}
		__syncthreads();
		for(uint8_t i = 0; i < moveCount; i++)
		{
			evalBlackMoveKernel(&results[i], boardTile, &moves[i], depth + 1, cornerTile);
		}

		__syncthreads();
		if(IS_ROOT_THREAD)
		{
			cudaFree(moves);
			//cudaFree(newBoard);
			resultVal = RESULT_RED_WIN;
		}
		__syncthreads();

		// Pick max result
		if(threadIdx.x < moveCount)
		{
			atomicMax(&resultVal, results[threadIdx.x]);
			__syncthreads();
			if(resultVal == results[threadIdx.x])
			{
				resultIndex = x;
			}
		}
	}
	else
	{
		// Create moves
		getRedMovesGPU(moves, moveCount, boardTile, cornerTile);

		// Evaluate Moves (recursive)
		if(IS_ROOT_THREAD)
		{
			gpuErrchk(cudaMalloc(&results, moveCount*sizeof(result_gpu_t)));
		}
		__syncthreads();
		for(uint8_t i = 0; i < moveCount; i++)
		{
			evalRedMoveKernel(&results[i], boardTile, &moves[i], depth + 1, cornerTile);
		}

		__syncthreads();
		if(IS_ROOT_THREAD)
		{
			cudaFree(moves);
			//cudaFree(newBoard);
			resultVal = RESULT_BLACK_WIN;
		}
		__syncthreads();

		// Pick min result
		if(threadIdx.x < moveCount)
		{
			atomicMin(&resultVal, results[threadIdx.x]);
			__syncthreads();
			if(resultVal == results[threadIdx.x])
			{
				resultIndex = x;
			}
		}
	}
	__syncthreads();
	if(IS_ROOT_THREAD)
	{
		if(depth == 0) printf("Result: %i\n", results[resultIndex]);
		*resultOut = results[resultIndex];
		cudaFree(results);
	}
}

__device__ void evalRedMoveKernel(result_gpu_t* resultOut, boardstate_t* board, Move* oldMoves, depth_t depth, boardpos_t (&cornerTile)[SQUARE_COUNT][4])
{
	//printf("Depth: %i\n", depth);
	__shared__ boardstate_t boardTile[SQUARE_COUNT];
	__shared__ Move moveTile;
	__shared__ unsigned int moveCount;
	__shared__ int resultVal;
	__shared__ int resultIndex;
	unsigned int x = threadIdx.x;

	// Copy board
	boardTile[x] = board[x];
	__syncthreads();

	// Execute Move (if root)
	if(IS_ROOT_THREAD)
	{
		moveTile = *oldMoves;
		boardTile[moveTile.newPos] = boardTile[moveTile.oldPos];
		boardTile[moveTile.oldPos] = SQUARE_EMPTY;
		if(MOVE_ISJUMP(moveTile)) boardTile[moveTile.jumpPos] = SQUARE_EMPTY;

		// Check for king
		if(moveTile.newPos < 4)
		{
			if(SQUARE_ISNOTEMPTY(boardTile[moveTile.newPos]))
			{
				boardTile[moveTile.newPos] |= 0x1;
			}
		}
		moveCount = 0;
		resultIndex = -1;
	}
	__syncthreads();

	// Check results and depth
	if(evalBoardSquareGPU(resultOut, boardTile, cornerTile)) return;
	if(depth == NODE_DEPTH_GPU)
	{
		printf("AHHHH\n");
		return;
	}

	__shared__ Move* moves;
	__shared__ result_gpu_t* results;
	//__shared__ boardstate_t* newBoard;
	__syncthreads();
	if(IS_ROOT_THREAD)
	{
		gpuErrchk(cudaMalloc(&moves, MOVE_BUFFER_SIZE*sizeof(Move)));
		//gpuErrchk(cudaMalloc(&newBoard, SQUARE_COUNT*sizeof(boardstate_t)));
	}
	//__syncthreads();
	//newBoard[x] = boardTile[x];
	//__syncthreads();
	if(moveTile.moveType == MOVE_JUMP_MULTI)
	{
		// Create moves
		getRedJumpsGPU(moves, moveCount, moveTile.newPos, boardTile, cornerTile);

		// Evaluate Moves (recursive)
		if(IS_ROOT_THREAD)
		{
			gpuErrchk(cudaMalloc(&results, moveCount*sizeof(result_gpu_t)));
		}
		__syncthreads();
		for(uint8_t i = 0; i < moveCount; i++)
		{
			evalRedMoveKernel(&results[i], boardTile, &moves[i], depth + 1, cornerTile);
		}

		__syncthreads();
		if(IS_ROOT_THREAD)
		{
			cudaFree(moves);
			//cudaFree(newBoard);
			resultVal = RESULT_BLACK_WIN;
		}
		__syncthreads();

		// Pick max result
		if(threadIdx.x < moveCount)
		{
			atomicMin(&resultVal, results[threadIdx.x]);
			__syncthreads();
			if(resultVal == results[threadIdx.x])
			{
				resultIndex = x;
			}
		}
	}
	else
	{
		// Create moves
		getBlackMovesGPU(moves, moveCount, boardTile, cornerTile);

		// Evaluate Moves (recursive)
		if(IS_ROOT_THREAD)
		{
			gpuErrchk(cudaMalloc(&results, moveCount*sizeof(result_gpu_t)));
		}
		__syncthreads();
		for(uint8_t i = 0; i < moveCount; i++)
		{
			evalBlackMoveKernel(&results[i], boardTile, &moves[i], depth + 1, cornerTile);
		}

		__syncthreads();
		if(IS_ROOT_THREAD)
		{
			cudaFree(moves);
			//cudaFree(newBoard);
			resultVal = RESULT_RED_WIN;
		}
		__syncthreads();

		// Pick min result
		if(threadIdx.x < moveCount)
		{
			atomicMax(&resultVal, results[threadIdx.x]);
			__syncthreads();
			if(resultVal == results[threadIdx.x])
			{
				resultIndex = x;
			}
		}
	}
	__syncthreads();
	if(IS_ROOT_THREAD)
	{
		*resultOut = results[resultIndex];
		cudaFree(results);
	}
}
__global__ void getResultsKernel(result_gpu_t* results_dev, boardstate_t* board_dev, Move* moves_dev)
{

	// Copy cornerTile for faster calculations
	__shared__ boardpos_t cornerTile[SQUARE_COUNT][4];
	unsigned int threadX = threadIdx.x;
	cornerTile[threadX][0] = cornerListDev[threadX][0];
	cornerTile[threadX][1] = cornerListDev[threadX][1];
	cornerTile[threadX][2] = cornerListDev[threadX][2];
	cornerTile[threadX][3] = cornerListDev[threadX][3];
	__syncthreads();

	unsigned int blockX = blockIdx.x;
	evalBlackMoveKernel(&results_dev[blockX], board_dev, &moves_dev[blockX], 0, cornerTile);
}

Move* GPUUtility::moves_dev = nullptr;
boardstate_t* GPUUtility::board_dev = nullptr;
result_gpu_t* GPUUtility::results_dev = nullptr;
boardpos_t previousMultiJumpPos = -1;

void GPUUtility::initializeGPU()
{
	// Allocate memory
	cudaMalloc(&moves_dev, sizeof(Move)*SQUARE_COUNT*4);
	cudaMalloc(&board_dev, sizeof(BoardState));
}

void GPUUtility::clear()
{
	// Free memory
	cudaFree(moves_dev);
	cudaFree(board_dev);
	cudaFree(results_dev);
}

Move GPUUtility::getMove(GameBoard board)
{
	// Generate initial moves
	std::vector<Move>* moves;
	if(previousMultiJumpPos == BOARD_POS_INVALID)
	{
		moves = AIUtility::getAllBlackMoves(board);
	}
	else
	{
		moves = AIUtility::getAllBlackJumps(board, previousMultiJumpPos);
	}

	if(moves->empty())
	{
		Move m;
		m.moveType = MOVE_INVALID;
		return m;
	}

	// Generate move tree
	std::vector<result_gpu_t> results_host;
	results_host.resize(moves->size());

	cudaMemcpy(board_dev, board.getBoardState(), sizeof(BoardState), cudaMemcpyHostToDevice);
	cudaMemcpy(moves_dev, moves->data(), sizeof(Move)*moves->size(), cudaMemcpyHostToDevice);
	getResultsKernel CUDA_KERNEL(static_cast<unsigned int>(moves->size()),32) (results_dev, board_dev, moves_dev);

	// Wait till finished;
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	cudaMemcpy(results_host.data(), results_dev, sizeof(result_gpu_t)*moves->size(), cudaMemcpyDeviceToHost);

	// Print moves & results
	for(uint8_t i = 0; i < moves->size(); i++)
	{
		Move m = moves->at(i);
		std::cout << "Move: " << +m.oldPos << "," << +m.jumpPos << "," << +m.newPos << "," << +m.moveType << " with result: " << +results_host.at(i) << '\n';
	}
	std::cout << std::endl;

	// Pick result
	auto iterator = std::max_element(std::begin(results_host), std::end(results_host));
	size_t a = std::distance(results_host.begin(), iterator);
	Move move = moves->at(a);

	// Check for multijump
	if(move.moveType == MOVE_JUMP_MULTI)
	{
		previousMultiJumpPos = move.newPos;
	}
	else previousMultiJumpPos = -1;

	return move;
}
