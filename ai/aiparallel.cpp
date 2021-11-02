#include "aiparallel.h"
#include "aiutility.h"

AIParallel::AIParallel(QObject *parent) : AI(parent)
{
	threadPool = new QThreadPool();
}

AIParallel::~AIParallel()
{
	delete threadPool;
}

void AIParallel::executeBlackMove(GameBoard board, Move &move, depth_t depth)
{
	// Execute Move
	board.move(move.oldPos, move.newPos);
	if(MOVE_ISJUMP(move)) board.setSquareState(move.jumpPos, SQUARE_EMPTY);

	// Check depth and evaluate
	//result_t result;
	//if(evalBoardResult(board, result)) return result;
	//else if(depth == NODE_DEPTH) return result;

	std::vector<Move>* moves;
	std::vector<result_t> results;
	size_t a;
	if(move.moveType == MOVE_JUMP_MULTI)
	{
		// Create moves
		moves = AIUtility::getAllBlackJumps(board, move.newPos);
		if(moves->empty())
		{
			moves = AIUtility::getAllBlackMoves(board);
		}

		// Evaluate Moves (recursive)
		for(uint8_t i = 0; i < moves->size(); i++)
		{
			//results.push_back(evalBlackMove(board, moves->at(i), depth + 1));
		}
		// Pick max result
		auto iterator = std::max_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	else
	{
		// Create moves
		moves = AIUtility::getAllRedMoves(board);

		// Evaluate Moves (recursive)
		for(uint8_t i = 0; i < moves->size(); i++)
		{
			//results.push_back(executeRedMove(board, moves->at(i), depth + 1));
		}
		// Pick min result
		auto iterator = std::min_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	//return results[a];
}

void AIParallel::executeRedMove(GameBoard board, Move &move, depth_t depth)
{
	// Execute Move
	board.move(move.oldPos, move.newPos);
	if(MOVE_ISJUMP(move)) board.setSquareState(move.jumpPos, SQUARE_EMPTY);

	// Check depth and evaluate
	//result_t result;
	//if(evalBoardResult(board, result)) return result;
	//else if(depth == NODE_DEPTH) return result;
	if(depth == NODE_DEPTH)
	{

	}
	else
	{
		std::vector<Move>* moves;
		std::vector<result_t> results;
		size_t a;
		if(move.moveType == MOVE_JUMP_MULTI)
		{
			// Create moves
			moves = AIUtility::getAllRedJumps(board, move.newPos);
			if(moves->empty())
			{
				moves = AIUtility::getAllRedMoves(board);
			}

			// Evaluate Moves (recursive)
			for(uint8_t i = 0; i < moves->size(); i++)
			{
				//results.push_back(evalRedMove(board, moves->at(i), depth + 1));
			}
		}
		else
		{
			// Create moves
			moves = AIUtility::getAllBlackMoves(board);

			// Evaluate Moves (recursive)
			for(uint8_t i = 0; i < moves->size(); i++)
			{
				//results.push_back(evalBlackMove(board, moves->at(i), depth + 1));
			}
		}
	}
}

Move AIParallel::getMove(GameBoard& board)
{
	std::vector<Move>* moves;
	if(previousMultiJumpPos == BOARD_POS_INVALID)
	{
		moves = AIUtility::getAllBlackMoves(board);
	}
	else
	{
		moves = AIUtility::getAllBlackJumps(board, previousMultiJumpPos);
		if(moves->empty())
		{
			moves = AIUtility::getAllBlackMoves(board);
		}
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
		//results.push_back(e(board, moves->at(i), 0));
	}

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
