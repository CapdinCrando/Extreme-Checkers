#ifndef AI_H
#define AI_H

#include <QObject>
#include "gameboard.h"
#include "defines.h"

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
	boardpos_t previousMultiJumpPos = -1;
};

#endif // AI_H
