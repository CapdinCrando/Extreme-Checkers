#include "aitask.h"
#include "aiutility.h"

void AITask::run()
{
	// Execute Move
	board.move(move.oldPos, move.newPos);
	if(MOVE_ISJUMP(move)) board.setSquareState(move.jumpPos, SQUARE_EMPTY);

	// Check depth and evaluate
	result_t result;
	if(AIUtility::evalBoardResult(board, result))
	{
		node->result = result;
		return;
	}
	else if(depth == NODE_DEPTH)
	{
		node->result = result;
		return;
	}

	std::vector<Move>* moves;
	if(node->isBlack)
	{
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
				Node* newNode = new Node;
				node->isBlack = true;
				node->children.push_back(newNode);
				QThreadPool::globalInstance()->start(new AITask(board, moves->at(i), depth + 1, newNode));
			}
		}
		else
		{
			// Create moves
			moves = AIUtility::getAllRedMoves(board);

			// Evaluate Moves (recursive)
			for(uint8_t i = 0; i < moves->size(); i++)
			{
				Node* newNode = new Node;
				node->isBlack = false;
				node->children.push_back(newNode);
				QThreadPool::globalInstance()->start(new AITask(board, moves->at(i), depth + 1, newNode));
			}
		}
	}
	else
	{
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
				Node* newNode = new Node;
				node->isBlack = false;
				node->children.push_back(newNode);
				QThreadPool::globalInstance()->start(new AITask(board, moves->at(i), depth + 1, newNode));
			}
		}
		else
		{
			// Create moves
			moves = AIUtility::getAllBlackMoves(board);

			// Evaluate Moves (recursive)
			for(uint8_t i = 0; i < moves->size(); i++)
			{
				Node* newNode = new Node;
				node->isBlack = true;
				node->children.push_back(newNode);
				QThreadPool::globalInstance()->start(new AITask(board, moves->at(i), depth + 1, newNode));
			}
		}
	}
}
