#include "aiminimax.h"
#include "aiutility.h"

Move AIMinimax::getMove(GameBoard& board)
{
	std::vector<Move>* moves;
	if(previousMultiJumpPos == BOARD_POS_INVALID)
	{
		moves = AIUtility::getAllBlackMoves(board);
	}
	else
	{
		moves = AIUtility::getAllBlackJumps(board, previousMultiJumpPos);
	}

	if(moves->empty())
	{
		Move m;
		m.moveType = MOVE_INVALID;
		return m;
	}

	std::vector<result_t> results;

	result_t alpha = RESULT_MIN;
	result_t beta = RESULT_MAX;
	result_t result = RESULT_MIN;
	for(uint8_t i = 0; i < moves->size(); i++)
	{
		result_t value = AIUtility::evalBlackMove(board, moves->at(i), 0, alpha, beta);
		results.push_back(value);
		result	= std::max(result, value);
		alpha = std::max(alpha, result);
		if(beta <= alpha) break;
	}

	// Pick result
	Move move = moves->at(AIUtility::selectResult(&results));

	// Check for multijump
	if(move.moveType == MOVE_JUMP_MULTI)
	{
		previousMultiJumpPos = move.newPos;
	}
	else previousMultiJumpPos = -1;

	return move;
}
