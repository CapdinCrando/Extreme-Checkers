#ifndef AIQUEUE_H
#define AIQUEUE_H

#include "ai.h"

class aiqueue : public AI
{
public:
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return "Level 4 - Parallel Queue AI"; }
};

#endif // AIQUEUE_H
