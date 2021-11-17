#ifndef AIPARALLEL_H
#define AIPARALLEL_H

#include <QThreadPool>

#include "ai.h"

class AIParallel : public AI
{
public:
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return "Level 3 - Parallel AI"; }
};

#endif // AIPARALLEL_H
