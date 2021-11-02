#ifndef AIPARALLEL_H
#define AIPARALLEL_H

#include <QThreadPool>

#include "ai.h"
#include "node.h"

class AIParallel : public AI
{
	Q_OBJECT
public:
	explicit AIParallel(QObject *parent = nullptr);
	~AIParallel();
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return "Level 3 - Parallel AI"; }

private:
};

#endif // AIPARALLEL_H
