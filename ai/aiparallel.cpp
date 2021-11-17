#include "aiparallel.h"
#include "aiutility.h"
#include "aitask.h"

#include <algorithm>

Move AIParallel::getMove(GameBoard& board)
{
	// Generate initial moves
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

	// Generate move tree
	std::vector<result_t> results;
	results.resize(moves->size());
	for(uint8_t i = 0; i < moves->size(); i++)
	{
		QThreadPool::globalInstance()->start(new AITask(board, moves->at(i), results.at(i)));
	}

	// Wait till finished
	QThreadPool::globalInstance()->waitForDone();

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
