#include "aiminimax.h"
#include "aiutility.h"

#include <iostream>

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

	for(uint8_t i = 0; i < moves->size(); i++)
	{
		results.push_back(AIUtility::evalBlackMove(board, moves->at(i), 0));
	}

	// Print moves & results
	for(uint8_t i = 0; i < moves->size(); i++)
	{
		Move m = moves->at(i);
		std::cout << "Move: " << +m.oldPos << "," << +m.jumpPos << "," << +m.newPos << "," << +m.moveType << " with result: " << +results.at(i) << '\n';
	}
	std::cout << std::endl;

	// Pick result
	auto iterator = std::max_element(std::begin(results), std::end(results));
	size_t a = std::distance(results.begin(), iterator);
	Move move = moves->at(a);

	// Check for multijump
	if(move.moveType == MOVE_JUMP_MULTI)
	{
		previousMultiJumpPos = move.newPos;
	}
	else previousMultiJumpPos = -1;

	return move;
}
