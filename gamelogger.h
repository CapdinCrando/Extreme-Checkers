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

public slots:
	void openLogFile(GameSettings settings);
	void logMessage(std::string message);

private:
	std::ofstream file;
};

#endif // GAMELOGGER_H
