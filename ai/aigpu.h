#ifndef AIGPU_H
#define AIGPU_H

#include "ai.h"

class AIGPU : public AI
{
public:
	AIGPU();
	~AIGPU();
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return "Level 4 - GPU AI"; }
};

#endif // AIGPU_H
