#include "aigpu.h"

#include "aiutility.h"
#include "gpuutility.h"

AIGPU::AIGPU()
{
	GPUUtility::initializeGPU();
}

Move AIGPU::getMove(GameBoard& board)
{
	return GPUUtility::getMove(board.getBoardState());
}
