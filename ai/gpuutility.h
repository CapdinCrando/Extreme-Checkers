#ifndef GPUUTILITY_H
#define GPUUTILITY_H

#include "../defines.h"
#include "../gameboard.h"

namespace GPUUtility
{
	Move getMove(BoardState* board);
	std::vector<Move>* getAllBlackMoves(BoardState* board);
}

#endif // GPUUTILITY_H
