#ifndef AIPARALLEL_H
#define AIPARALLEL_H

#include "ai.h"

#define NODE_DEPTH 6

class AIParallel : public AI
{
	Q_OBJECT
public:
	explicit AIParallel(QObject *parent = nullptr) : AI(parent) {};
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return "Level 3 - Parallel AI"; }

private:
	std::vector<Move>* getAllRedMoves(GameBoard &board);
	std::vector<Move>* getAllRedJumps(GameBoard &board, boardpos_t pos);
	bool evalBoardResult(GameBoard &board, result_t& resultOut);
	result_t evalBlackMove(GameBoard board, Move& move, depth_t depth);
	result_t evalRedMove(GameBoard board, Move& move, depth_t depth);
};

#endif // AIPARALLEL_H
