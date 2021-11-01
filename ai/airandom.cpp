#include "airandom.h"

Move AIRandom::getMove(GameBoard& board)
{
	std::vector<Move>* moves;
	if(previousMultiJumpPos == BOARD_POS_INVALID)
	{
		moves = getAllBlackMoves(board);
	}
	else
	{
		moves = getAllBlackJumps(board, previousMultiJumpPos);
		if(moves->empty())
		{
			moves = getAllBlackMoves(board);
		}
	}

	Move move;
	move.moveType = MOVE_INVALID;
	if(!moves->empty())
	{
		move = moves->at(rand() % moves->size());
	}

	if(move.moveType == MOVE_JUMP_MULTI)
	{
		previousMultiJumpPos = move.newPos;
	}
	else previousMultiJumpPos = -1;

	delete moves;
	return move;
}


