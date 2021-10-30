#include "airandom.h"

Move AIRandom::getMove(GameBoard& board)
{
	std::vector<Move>* moves = getAllBlackMoves(board);

	Move move = moves->at(rand() % moves->size());
	delete moves;
	return move;
}


