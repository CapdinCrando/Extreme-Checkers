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
	AIMove getMove(GameBoard& board) override;
	QString getDescription() override { return description; }

protected:
	const QString description = "Level 2 - Minimax AI";

private:
	std::vector<AIMove>* getAllBlackMoves(GameBoard &board);
	std::vector<AIMove>* getAllRedMoves(GameBoard &board);
	std::vector<AIMove>* getAllBlackJumps(GameBoard &board, boardpos_t pos);
	std::vector<AIMove>* getAllRedJumps(GameBoard &board, boardpos_t pos);
	bool evalBoardResult(GameBoard &board, result_t& resultOut);
	result_t evalBlackMove(GameBoard board, AIMove& move, depth_t depth);
	result_t evalRedMove(GameBoard board, AIMove& move, depth_t depth);
};

#endif // AIMINIMAX_H
