#ifndef GAMELOGGER_H
#define GAMELOGGER_H

#include <fstream>

#include <QObject>

#include "defines.h"

class GameLogger : public QObject
{
	Q_OBJECT

public:
	GameLogger() : QObject() {};
	~GameLogger();
	void openLogFile(GameSettings settings);

public slots:
	void logRedMove(Move move);
	void logBlackMove(Move move, long long moveDurationMicro);
	void logGameOver(GameState gameState);

private:
	std::ofstream file;
};

#endif // GAMELOGGER_H
