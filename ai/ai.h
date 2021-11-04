#ifndef AI_H
#define AI_H

#include "gameboard.h"
#include "defines.h"
#include <QStringList>

class AI
{
public:
	virtual ~AI() {};
	virtual Move getMove(GameBoard& board) = 0;
	virtual QString getDescription() = 0;

protected:
	boardpos_t previousMultiJumpPos = -1;
};

#endif // AI_H
