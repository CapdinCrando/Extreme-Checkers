#ifndef AIRANDOM_H
#define AIRANDOM_H

#include "ai.h"

class AIRandom : public AI
{
	Q_OBJECT
public:
	explicit AIRandom(QObject *parent = nullptr) : AI(parent) {};
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return "Level 1 - Random AI"; }
};

#endif // AIRANDOM_H
