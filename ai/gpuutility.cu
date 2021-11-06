#include "gpuutility.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdio.h>

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__device__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
	  printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

typedef int result_gpu_t;
#define IS_ROOT_THREAD threadIdx.x == 0
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
		if(*resultOut < -3)
		{
			//printf("Eval: %i,%i,%i,%i,%i,%i ; %i\n", *resultOut, blackCount, redCount, blackMoveFound, redMoveFound, retVal, board[0]);
		}
	}
	__syncthreads();
	return retVal;
}

// Definition
__global__ void evalRedMoveKernel(result_gpu_t* result, boardstate_t* board, Move* oldMoves, depth_t depth);

__global__ void evalBlackMoveKernel(result_gpu_t* result, boardstate_t* board, Move* oldMoves, depth_t depth)
{
	__shared__ boardpos_t cornerTile[SQUARE_COUNT][4];
	__shared__ boardstate_t boardTile[SQUARE_COUNT];
	__shared__ Move moveTile;
	__shared__ unsigned int moveCount;
	__shared__ int resultVal;
	__shared__ int resultIndex;
	result_gpu_t* resultOut = &result[blockIdx.x];
	unsigned int x = threadIdx.x;
	//if(IS_ROOT_THREAD) if(board[0] == 0 && board[1] == 0) printf("Black board empty @D%i, X:%i\n", depth, oldMoves[blockIdx.x].newPos);

	// Copy board
	boardTile[x] = board[x];

	// Copy cornerTile for faster calculations
	cornerTile[x][0] = cornerListDev[x][0];
	cornerTile[x][1] = cornerListDev[x][1];
	cornerTile[x][2] = cornerListDev[x][2];
	cornerTile[x][3] = cornerListDev[x][3];

	// Execute Move (if root)
	if(IS_ROOT_THREAD)
	{
		moveTile = oldMoves[blockIdx.x];
		boardTile[moveTile.newPos] = boardTile[moveTile.oldPos];
		boardTile[moveTile.oldPos] = SQUARE_EMPTY;
		if(MOVE_ISJUMP(moveTile)) boardTile[moveTile.jumpPos] = SQUARE_EMPTY;

		// Check for king
		if(moveTile.newPos > 27)
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

	// Check depth
	if(evalBoardSquareGPU(resultOut, boardTile, cornerTile))
	{
		if(IS_ROOT_THREAD)
		{
			//if(*resultOut < -3) printf("Black result @D%i: %i\n", depth, *resultOut);
		}
		return;
	}
	//if(IS_ROOT_THREAD) printf("Result @ D%i: %i\n", depth, *result);
	if(depth == NODE_DEPTH_GPU)
	{
		if(IS_ROOT_THREAD)
		{
			//printf("Red result @D%i: %i\n", depth, *resultOut);
		}
		return;
	}

	__shared__ Move* moves;
	__shared__ result_gpu_t* results;
	__shared__ boardstate_t* newBoard;
	if(IS_ROOT_THREAD)
	{
		gpuErrchk(cudaMalloc(&moves, MOVE_BUFFER_SIZE*sizeof(Move)));
		gpuErrchk(cudaMalloc(&newBoard, SQUARE_COUNT*sizeof(boardstate_t)));
	}
	__syncthreads();
	newBoard[x] = boardTile[x];
	__syncthreads();

	if(moveTile.moveType == MOVE_JUMP_MULTI)
	{
		// Create moves
		getBlackJumpsGPU(moves, moveCount, moveTile.newPos, boardTile, cornerTile);
		if(moveCount == 0)
		{
			getBlackMovesGPU(moves, moveCount, boardTile, cornerTile);
		}

		// Evaluate Moves (recursive)
		__syncthreads();
		/*if(moveCount == 0)
		{
			if(IS_ROOT_THREAD) //printf("Black result1 @D%i: %i ; %i\n", depth, *resultOut, boardTile[0]);
			return;
		}*/
		if(IS_ROOT_THREAD)
		{
			gpuErrchk(cudaMalloc(&results, moveCount*sizeof(result_gpu_t)));
			//if(depth == 3) if(newBoard[0] == 0 && newBoard[1] == 0) printf("Null found\n");
			evalBlackMoveKernel CUDA_KERNEL(moveCount, SQUARE_COUNT) (results, newBoard, moves, depth + 1);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			cudaFree(moves);
			cudaFree(newBoard);
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
		//printf("Possible Move: %i,%i,%i,%i\n", moves[x].oldPos, moves[x].newPos, moves[x].jumpPos, moves[x].moveType);
		__syncthreads();
		/*if(moveCount == 0)
		{
			if(IS_ROOT_THREAD) //printf("Black result2 @D%i: %i ; %i,%i,%i\n", depth, *resultOut, board[0], boardTile[0], newBoard[0]);
			return;
		}*/
		if(IS_ROOT_THREAD)
		{
			gpuErrchk(cudaMalloc(&results, moveCount*sizeof(result_gpu_t)));
			//if(newBoard == nullptr) printf("Null found\n");
			evalRedMoveKernel CUDA_KERNEL(moveCount, SQUARE_COUNT) (results, newBoard, moves, depth + 1);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			cudaFree(moves);
			cudaFree(newBoard);
			resultVal = RESULT_BLACK_WIN;
		}
		__syncthreads();

		// Pick min result
		if(threadIdx.x < moveCount)
		{
			//printf("Result from Red: %i\n", results[threadIdx.x]);
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
		*resultOut = results[resultIndex];
		cudaFree(results);
		//printf("Black Kernel Finished @D%i\n", depth);
	}
}

__global__ void evalRedMoveKernel(result_gpu_t* result, boardstate_t* board, Move* oldMoves, depth_t depth)
{
	__shared__ boardpos_t cornerTile[SQUARE_COUNT][4];
	__shared__ boardstate_t boardTile[SQUARE_COUNT];
	__shared__ Move moveTile;
	__shared__ unsigned int moveCount;
	__shared__ int resultVal;
	__shared__ int resultIndex;
	result_gpu_t* resultOut = &result[blockIdx.x];
	unsigned int x = threadIdx.x;
	//if(threadIdx.x == 1) if(board[0] == 0 && board[1] == 0) printf("Red board empty @D%i\n", depth);

	// Copy board
	boardTile[x] = board[x];
	__syncthreads();
	//if(threadIdx.x == 1) if(boardTile[0] == 0 && boardTile[1] == 0) printf("Red boardtile empty @D%i\n", depth);

	// Copy cornerTile for faster calculations
	cornerTile[x][0] = cornerListDev[x][0];
	cornerTile[x][1] = cornerListDev[x][1];
	cornerTile[x][2] = cornerListDev[x][2];
	cornerTile[x][3] = cornerListDev[x][3];

	// Execute Move (if root)
	if(IS_ROOT_THREAD)
	{
		moveTile = oldMoves[blockIdx.x];
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

	// Check depth
	if(evalBoardSquareGPU(resultOut, boardTile, cornerTile))
	{
		if(IS_ROOT_THREAD)
		{
			//if(*resultOut < -3) printf("Red result @D%i: %i\n", depth, *resultOut);
		}
		return;
	}
	if(depth == NODE_DEPTH_GPU)
	{
		if(IS_ROOT_THREAD)
		{
			//printf("Red result @D%i: %i\n", depth, *resultOut);
		}
		return;
	}

	__shared__ Move* moves;
	__shared__ result_gpu_t* results;
	__shared__ boardstate_t* newBoard;
	__syncthreads();
	if(IS_ROOT_THREAD)
	{
		gpuErrchk(cudaMalloc(&moves, MOVE_BUFFER_SIZE*sizeof(Move)));
		gpuErrchk(cudaMalloc(&newBoard, SQUARE_COUNT*sizeof(boardstate_t)));
	}
	__syncthreads();
	newBoard[x] = boardTile[x];
	__syncthreads();
	if(moveTile.moveType == MOVE_JUMP_MULTI)
	{
		// Create moves
		getRedJumpsGPU(moves, moveCount, moveTile.newPos, boardTile, cornerTile);
		if(moveCount == 0)
		{
			getRedMovesGPU(moves, moveCount, boardTile, cornerTile);
		}

		// Evaluate Moves (recursive)
		__syncthreads();
		/*if(moveCount == 0)
		{
			if(IS_ROOT_THREAD) printf("Red result @D%i: %i ; %i\n", depth, *resultOut, boardTile[0]);
			return;
		}*/
		if(IS_ROOT_THREAD)
		{
			gpuErrchk(cudaMalloc(&results, moveCount*sizeof(result_gpu_t)));
			//if(newBoard == nullptr) printf("Null found\n");
			evalRedMoveKernel CUDA_KERNEL(moveCount, SQUARE_COUNT) (results, newBoard, moves, depth + 1);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			cudaFree(moves);
			cudaFree(newBoard);
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
		//if(threadIdx.x == 1) if(newBoard[0] == 0 && newBoard[1] == 0) printf("Red newboard empty @D%i\n", depth);

		// Evaluate Moves (recursive)
		__syncthreads();
		/*if(moveCount == 0)
		{
			//if(IS_ROOT_THREAD) printf("Red result @D%i: %i\n", depth, *resultOut);
			return;
		}*/
		if(IS_ROOT_THREAD)
		{
			gpuErrchk(cudaMalloc(&results, moveCount*sizeof(result_gpu_t)));
			//if(depth == 3) if(newBoard[0] == 0 && newBoard[1] == 0) printf("Null found2\n");
			evalBlackMoveKernel CUDA_KERNEL(moveCount, SQUARE_COUNT) (results, newBoard, moves, depth + 1);
			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk( cudaDeviceSynchronize() );
			cudaFree(moves);
			cudaFree(newBoard);
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
		//printf("Red Kernel Finished @D%i\n", depth);
	}
}

__global__ void getMoveKernel(Move* move, boardstate_t* board)
{
	__shared__ Move moveTile;
	__shared__ boardpos_t cornerTile[SQUARE_COUNT][4];
	__shared__ int maxResult;
	__shared__ uint16_t maxIndex;
	__shared__ unsigned int moveCount;
	__shared__ Move* moves;
	if(IS_ROOT_THREAD)
	{
		moveCount = 0;
		cudaMalloc(&moves, MOVE_BUFFER_SIZE*sizeof(Move));
	}
	__syncthreads();

	unsigned int x = threadIdx.x;

	// Copy cornerTile for faster calculations
	cornerTile[x][0] = cornerListDev[x][0];
	cornerTile[x][1] = cornerListDev[x][1];
	cornerTile[x][2] = cornerListDev[x][2];
	cornerTile[x][3] = cornerListDev[x][3];
	__syncthreads();

	if(previousMultiJumpPosGPU == BOARD_POS_INVALID)
	{
		getBlackMovesGPU(moves, moveCount, board, cornerTile);
	}
	else
	{
		getBlackJumpsGPU(moves, moveCount, previousMultiJumpPosGPU, board, cornerTile);
		if(moveCount == 0)
		{
			getBlackMovesGPU(moves, moveCount, board, cornerTile);
		}
	}

	if(moveCount == 0)
	{
		if(IS_ROOT_THREAD)
		{
			moveTile.moveType = MOVE_INVALID;
			*move = moveTile;
		}
		return;
	}

	__shared__ result_gpu_t* results;
	if(IS_ROOT_THREAD)
	{
		//moveCount = 1;
		cudaMalloc(&results, moveCount*sizeof(result_gpu_t));
		//printf("Possible move count: %i\n", moveCount);
		evalBlackMoveKernel CUDA_KERNEL(moveCount, SQUARE_COUNT) (results, board, moves, 0);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		//printf("Child Kernels finished\n");
		maxResult = RESULT_RED_WIN;
		maxIndex = 0;
	}
	__syncthreads();

	// Pick result
	if(x < moveCount)
	{
		//printf("Possible Move: %i,%i,%i,%i with a result of %i\n", moves[x].oldPos, moves[x].newPos, moves[x].jumpPos, moves[x].moveType, results[x]);
		atomicMax(&maxResult, results[x]);
		__syncthreads();
		if(maxResult == results[x])
		{
			maxIndex = x;
		}
		__syncthreads();
	}
	__syncthreads();
	if(IS_ROOT_THREAD)
	{
		moveTile = moves[maxIndex];

		// Check for multijump
		if(moveTile.moveType == MOVE_JUMP_MULTI)
		{
			previousMultiJumpPosGPU = moveTile.newPos;
		}
		else previousMultiJumpPosGPU = -1;
		*move = moveTile;
		//printf("Selected Move: %i,%i,%i,%i with a result of %i\n", moveTile.oldPos, moveTile.newPos, moveTile.jumpPos, moveTile.moveType, maxResult);
	}
	__syncthreads();
}

void GPUUtility::initializeGPU()
{
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, NODE_DEPTH_GPU + 1);
	//cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 2048);
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 67108864);
	//cudaDeviceSetLimit(cudaLimitStackSize, 1024);
	size_t stack_limit;
	cudaDeviceGetLimit(&stack_limit,cudaLimitStackSize);
	printf("Stack size: %llu\n", stack_limit);
	size_t heap_limit;
	cudaDeviceGetLimit(&heap_limit,cudaLimitMallocHeapSize);
	printf("Heap size: %llu\n", heap_limit);
}

Move GPUUtility::getMove(BoardState* board)
{	
	Move *move_host, *move_dev;
	move_host = new Move;
	cudaMalloc(&move_dev, sizeof(Move));

	boardstate_t *board_dev;
	cudaMalloc(&board_dev, sizeof(BoardState));
	cudaMemcpy(board_dev, board, sizeof(BoardState), cudaMemcpyHostToDevice);

	getMoveKernel CUDA_KERNEL(1,32) (move_dev, board_dev);
	cudaDeviceSynchronize();
	cudaMemcpy(move_host, move_dev, sizeof(Move), cudaMemcpyDeviceToHost);
	return *move_host;

	// You forgot to free memory dummy
}
