#ifndef AI_H
#define AI_H

#include <QObject>
#include "gameboard.h"
#include "defines.h"

enum MoveTypes : unsigned char
{
	MOVE_INVALID = 0,
	MOVE_MOVE = 1,
	MOVE_JUMP = 2,
	MOVE_JUMP_MULTI = 3,
};

struct AIMove
{
	char newPos : 6;
	char oldPos : 6;
	char jumpPos : 6;
	unsigned char moveType : 2;
};

typedef uint8_t depth_t;
typedef signed char result_t;

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
	virtual AIMove getMove(GameBoard& board) = 0;
	virtual QString getDescription() = 0;
};

#endif // AI_H
