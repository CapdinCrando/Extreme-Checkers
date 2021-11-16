#include "aiparallel.h"
#include "aiutility.h"
#include "aitask.h"

#include <algorithm>

result_t minimax(depth_t depth, Node* node)
{
	if(node->children.empty()) return node->result;

	std::vector<result_t> results;
	for(uint8_t i = 0; i < node->children.size(); i++)
	{
		results.push_back(minimax(depth + 1, node->children.at(i)));
	}
	size_t a;
	if(node->isBlack)
	{
		auto iterator = std::max_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	else
	{
		auto iterator = std::min_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	return results[a];
}

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
	std::vector<Node*> children;
	for(uint8_t i = 0; i < moves->size(); i++)
	{
		Node* newNode = new Node;
		newNode->isBlack = true;
		children.push_back(newNode);
		QThreadPool::globalInstance()->start(new AITask(board, moves->at(i), 0, newNode));
	}

	// Wait till finished
	QThreadPool::globalInstance()->waitForDone();

	// Execute Minimax Algorithm
	std::vector<result_t> results;
	for(uint8_t i = 0; i < children.size(); i++)
	{
		results.push_back(minimax(0, children.at(i)));
	}

	// Free memory
	for(uint8_t i = 0; i < children.size(); i++)
	{
		delete children.at(i);
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
