#ifndef AI_H
#define AI_H

#include <QObject>
#include "gameboard.h"
#include "defines.h"

typedef uint8_t depth_t;

enum ResultType : result_t
{
	RESULT_TIE = -13,
	RESULT_RED_WIN = -20,
	RESULT_BLACK_WIN = 20
};

struct Result
{
	signed char value : 5;
	result_t type : 3;
};

class AI : public QObject
{
	Q_OBJECT
public:
	explicit AI(QObject *parent = nullptr) : QObject(parent) {};
	virtual Move getMove(GameBoard& board) = 0;
	virtual QString getDescription() = 0;

protected:
	std::vector<Move>* getAllBlackMoves(GameBoard &board);
	std::vector<Move>* getAllBlackJumps(GameBoard &board, boardpos_t pos);
	boardpos_t previousMultiJumpPos = -1;
};

#endif // AI_H
