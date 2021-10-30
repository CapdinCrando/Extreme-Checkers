#include "aiminimax.h"

#define NODE_DEPTH 5

result_t evalBoardState(BoardState &board)
{

}

result_t evalBlackMove(BoardState& board, depth_t depth)
{
	if(depth == NODE_DEPTH) return evalBoardState(board);

	std::vector<MoveTreeNode> children;
	// Create moves
}

AIMove AIMinimax::getMove(BoardState& board)
{
	Q_UNUSED(board);

	// getAllMoves(board)
	// create list of values
	// evaluateBlackMove
	//		checkDepth
	//		evaluateRedMove


	return AIMove();
}


