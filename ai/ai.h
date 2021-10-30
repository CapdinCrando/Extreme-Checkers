#ifndef AI_H
#define AI_H

#include <QObject>
#include "gameboard.h"

enum MoveTypes : unsigned char
{
	MOVE_INVALID = 0,
	MOVE_MOVE = 1,
	MOVE_JUMP = 2,
	MOVE_JUMP_MULTI = 3,
};

struct AIMove
{
	unsigned char newPos : 5;
	unsigned char oldPos : 5;
	unsigned char jumpPos : 5;
	unsigned char jumpType : 3;
};

typedef uint8_t depth_t;
typedef int8_t result_t;

enum ResultType : result_t
{
	RESULT_RED_WIN = -20,
	RESULT_TIE = 0,
	RESULT_BLACK_WIN = 20
};

class AI : public QObject
{
	Q_OBJECT
public:
	explicit AI(QObject *parent = nullptr) : QObject(parent) {};
	virtual AIMove getMove(BoardState& board) = 0;
	virtual QString getDescription() = 0;
};

#endif // AI_H
