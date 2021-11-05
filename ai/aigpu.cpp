#include "aigpu.h"

#include "aiutility.h"
#include "gpuutility.h"

Move AIGPU::getMove(GameBoard& board)
{
	std::vector<Move>* moves;
	BoardState* boardState = board.getBoardState();
	if(previousMultiJumpPos == BOARD_POS_INVALID)
	{
		moves = GPUUtility::getAllBlackMoves(boardState);
	}
	else
	{
		moves = AIUtility::getAllBlackJumps(board, previousMultiJumpPos);
		if(moves->empty())
		{
			moves = GPUUtility::getAllBlackMoves(boardState);
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
