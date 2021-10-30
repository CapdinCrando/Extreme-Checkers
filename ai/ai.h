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

struct AIMove {
	unsigned char newPos : 5;
	unsigned char oldPos : 5;
	unsigned char jumpPos : 5;
	unsigned char jumpType : 3;
};

class AI : public QObject
{
	Q_OBJECT
public:
	explicit AI(QObject *parent = nullptr);
	virtual AIMove getMove() = 0;

signals:

};

#endif // AI_H
