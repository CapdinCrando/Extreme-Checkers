#ifndef AIMINIMAX_H
#define AIMINIMAX_H

#include "ai.h"

class AIMinimax : public AI
{
public:
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return "Level 2 - Minimax AI"; }

private:
	result_t evalBlackMove(GameBoard board, Move& move, depth_t depth);
	result_t evalRedMove(GameBoard board, Move& move, depth_t depth);
};

#endif // AIMINIMAX_H
