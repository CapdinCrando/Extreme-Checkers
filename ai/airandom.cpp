#include "airandom.h"

Move AIRandom::getMove(GameBoard& board)
{
	Move move;

	std::vector<Move>* moves = getAllBlackMoves(board);
	if(!moves->empty())
	{
		move = moves->at(rand() % moves->size());
	}

	delete moves;
	return move;
}


