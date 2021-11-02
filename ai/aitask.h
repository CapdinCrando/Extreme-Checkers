#ifndef AITASK_H
#define AITASK_H

#include <QRunnable>

#include "ai.h"
#include "node.h"
#include "gameboard.h"

#define NODE_DEPTH 6

class AITask : public QRunnable
{

public:
	AITask(GameBoard& board, Move& m, depth_t depth, Node* node);
	void run();

private:
	GameBoard board;
	Move m;
	depth_t depth;
	Node* node;

	std::vector<Move>* getAllRedMoves(GameBoard &board);
	std::vector<Move>* getAllRedJumps(GameBoard &board, boardpos_t pos);
	bool evalBoardResult(GameBoard &board, result_t& resultOut);
	void executeBlackMove(GameBoard board, Move& move, depth_t depth);
	void executeRedMove(GameBoard board, Move& move, depth_t depth);
};

#endif // AITASK_H
