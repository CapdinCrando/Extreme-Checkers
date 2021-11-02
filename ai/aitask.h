#ifndef AITASK_H
#define AITASK_H

#include <QRunnable>
#include <QThreadPool>

#include "node.h"
#include "gameboard.h"

#define NODE_DEPTH 6

#define NODE_DEPTH_CHECK NODE_DEPTH-1

class AITask : public QRunnable
{
public:
	AITask(GameBoard& board, Move& move, depth_t depth, Node* node) :
		board(board), move(move), depth(depth), node(node) {};
	void run() override;

private:
	GameBoard board;
	Move move;
	depth_t depth;
	Node* node;
};

#endif // AITASK_H
