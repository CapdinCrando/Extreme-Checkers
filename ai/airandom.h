#ifndef AIRANDOM_H
#define AIRANDOM_H

#include "ai.h"

class AIRandom : public AI
{
	Q_OBJECT
public:
	explicit AIRandom(QObject *parent = nullptr);
	AIMove getMove() override;
};

#endif // AIRANDOM_H
