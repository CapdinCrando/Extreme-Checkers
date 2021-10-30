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
	Move getMove(GameBoard& board) override;
	QString getDescription() override { return description; }

protected:
	const QString description = "Level 2 - Minimax AI";

private:
	std::vector<Move>* getAllRedMoves(GameBoard &board);
	std::vector<Move>* getAllBlackJumps(GameBoard &board, boardpos_t pos);
	std::vector<Move>* getAllRedJumps(GameBoard &board, boardpos_t pos);
	bool evalBoardResult(GameBoard &board, result_t& resultOut);
	result_t evalBlackMove(GameBoard board, Move& move, depth_t depth);
	result_t evalRedMove(GameBoard board, Move& move, depth_t depth);
};

#endif // AIMINIMAX_H
