#ifndef GAMEENGINE_H
#define GAMEENGINE_H

#include <QObject>
#include "gameboard.h"

class GameEngine : QObject
{
	Q_OBJECT

public:
	GameEngine();
	~GameEngine();

	void resetGame();
	void move(boardpos_t pos1, boardpos_t pos2);
	void getPossibleMoves(boardpos_t pos);
	SquareState getSquareState(boardpos_t index);

signals:
	void updateDisplay(BoardState b);


private:
	GameBoard gameBoard;
};

#endif // GAMEENGINE_H
