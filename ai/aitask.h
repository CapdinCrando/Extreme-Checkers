#ifndef AITASK_H
#define AITASK_H

#include <QRunnable>
#include <QThreadPool>

#include "node.h"
#include "gameboard.h"

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
