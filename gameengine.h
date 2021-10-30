#ifndef GAMEENGINE_H
#define GAMEENGINE_H

#include <QObject>
#include "gameboard.h"
#include "defines.h"
#include "ai/aimanager.h"



class GameEngine : public QObject
{
	Q_OBJECT

public:
	GameEngine();
	~GameEngine();

	void resetGame();
	std::vector<Move> getRedMoves(boardpos_t pos);
	SquareState getSquareState(boardpos_t index);
	void executeRedMove(Move move);
	void saveSettings(GameSettings settings);

signals:
	void displayMove(Move move, bool kingPiece);
	void displayMultiJump(std::vector<Move> moves, SquareState checkerType);
	void blackMoveFinished();
	void gameOver(GameState gameState);

private:
	void move(Move move);
	void executeBlackMove();
	bool checkForKing(boardpos_t pos);
	bool checkRedWin();
	bool checkBlackWin();
	void checkRedTie();
	void checkBlackTie();
	bool checkIfRedMoveExists();
	std::vector<Move> getAllBlackMoves();
	Move getAIMove();

	AIManager* aiManager;

	GameBoard gameBoard;
	bool jumpExists = false;
};

#endif // GAMEENGINE_H
