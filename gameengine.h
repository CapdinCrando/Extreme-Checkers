#ifndef GAMEENGINE_H
#define GAMEENGINE_H

#include "gameboard.h"
#include "defines.h"
#include "ai/aimanager.h"
#include <gamelogger.h>

#include <QObject>
#include <QTimer>

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
	void executeBlackMove();
	void logRedMove(Move move);
	void logBlackMove(Move move, long long moveDurationMicro);

private slots:
	void calculateMove();

private:
	bool checkGameOver(bool isBlackTurn);
	void handleBlackMove(Move move);
	void move(Move move);
	bool canBlackMove();
	Move getAIMove();

	AIManager* aiManager;
	GameLogger* logger;
	GameSettings settings;

	GameBoard gameBoard;
	bool jumpExists = false;
};

#endif // GAMEENGINE_H
