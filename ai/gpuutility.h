#ifndef GPUUTILITY_H
#define GPUUTILITY_H

#include "../defines.h"
#include "../gameboard.h"

namespace GPUUtility
{
	void testPrint();
	std::vector<Move>* getAllBlackMoves(BoardState* board);
}

#endif // GPUUTILITY_H
