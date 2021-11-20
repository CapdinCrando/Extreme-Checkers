#ifndef AITASK_H
#define AITASK_H

#include <QRunnable>
#include <QThreadPool>

#include "defines.h"
#include "gameboard.h"

class AITask : public QRunnable
{
public:
	AITask(GameBoard& board, Move& move, result_t& resultOut) :
		board(board), move(move), resultOut(resultOut) {};
	void run() override;

private:
	GameBoard board;
	Move move;
	result_t& resultOut;
};

#endif // AITASK_H
