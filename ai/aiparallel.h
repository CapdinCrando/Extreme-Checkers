#ifndef AIPARALLEL_H
#define AIPARALLEL_H

#include "ai.h"
#include <QThreadPool>

#define NODE_DEPTH 6

struct Node
{
	GameBoard board;
	Move m;
};

class AIParallel : public AI
{
	Q_OBJECT
public:
	explicit AIParallel(QObject *parent = nullptr);
	~AIParallel();
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return "Level 3 - Parallel AI"; }

private:
	QThreadPool* threadPool;
};

#endif // AIPARALLEL_H
