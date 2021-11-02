#include "aiparallel.h"
#include "aiutility.h"
#include "aitask.h"

#include <algorithm>

AIParallel::AIParallel(QObject *parent) : AI(parent)
{

}

AIParallel::~AIParallel()
{

}

result_t minimax(depth_t depth, Node* node, result_t alpha, result_t beta)
{
	if(node->children.empty()) return node->result;
	if(depth == NODE_DEPTH) return node->result;

	if(node->isBlack)
	{
		result_t best = RESULT_BLACK_WIN;
		for(uint8_t i = 0; i < node->children.size(); i++)
		{
			result_t val = minimax(depth + 1, node->children.at(i), alpha, beta);
			best = std::min(best, val);
			beta = std::min(beta, best);

			if(beta <= alpha) break;
		}
		return best;
	}
	else
	{
		result_t best = RESULT_RED_WIN;
		for(uint8_t i = 0; i < node->children.size(); i++)
		{
			result_t val = minimax(depth + 1, node->children.at(i), alpha, beta);
			best = std::max(best, val);
			alpha = std::max(alpha, best);

			if(beta <= alpha) break;
		}
		return best;
	}
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
	result_t best = RESULT_BLACK_WIN;
	result_t alpha = RESULT_RED_WIN;
	result_t beta = RESULT_BLACK_WIN;
	std::vector<result_t> results;
	for(uint8_t i = 0; i < children.size(); i++)
	{
		result_t result = minimax(0, children.at(i), alpha, beta);
		results.push_back(result);
		best = std::min(best, result);
		beta = std::min(beta, best);

		if(beta <= alpha) break;
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
