#ifndef GPUUTILITY_H
#define GPUUTILITY_H

#include "../defines.h"
#include "../gameboard.h"

namespace GPUUtility
{
	void initializeGPU();
	Move getMove(BoardState* board);
}

#endif // GPUUTILITY_H
