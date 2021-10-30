#ifndef AIMINIMAX_H
#define AIMINIMAX_H

#include "ai.h"

struct MoveTreeNode
{
	std::vector<MoveTreeNode*> children;
};

class AIMinimax : public AI
{
	Q_OBJECT
public:
	explicit AIMinimax(QObject *parent = nullptr) : AI(parent) {};
	AIMove getMove(BoardState& board) override;
	QString getDescription() override { return description; }

protected:
	const QString description = "Level 2 - Minimax AI";
};

#endif // AIMINIMAX_H
