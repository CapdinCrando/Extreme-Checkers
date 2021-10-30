#ifndef AIRANDOM_H
#define AIRANDOM_H

#include "ai.h"

class AIRandom : public AI
{
	Q_OBJECT
public:
	explicit AIRandom(QObject *parent = nullptr);
	AIMove getMove() override;
	QString getDescription() override { return description; }

protected:
	const QString description = "Level 1 - Random AI";
};

#endif // AIRANDOM_H
