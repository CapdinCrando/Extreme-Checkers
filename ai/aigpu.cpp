#include "aigpu.h"

#include "aiutility.h"
#include "gpuutility.h"

Move AIGPU::getMove(GameBoard& board)
{
	return GPUUtility::getMove();
}
