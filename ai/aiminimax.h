#ifndef AIMINIMAX_H
#define AIMINIMAX_H

#include "ai.h"

class AIMinimax : public AI
{
public:
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return "Level 2 - Minimax AI"; }
};

#endif // AIMINIMAX_H
