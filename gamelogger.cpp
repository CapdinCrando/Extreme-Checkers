#include "gamelogger.h"

#include <ctime>
#include <iomanip>
#include <string>

GameLogger::~GameLogger()
{
	if(file.is_open()) file.close();
}

void GameLogger::openLogFile(GameSettings settings)
{
	if(file.is_open()) file.close();

	if(settings.enableLogging)
	{
		std::string s;
		std::time_t rawTime = std::time(nullptr);
		std::tm localTime;
		localtime_s(&localTime, &rawTime);

		char logName[] = "log_YYYY-MM-DDT-HH-MM-SS.csv";
		std::strftime(&logName[4], sizeof(logName), "%Y-%m-%dT%H-%M-%S.csv", &localTime);

		file.open(logName, std::ios::out);

		file << "============================= GAME INFORMATION =============================\n" <<
				"File Name:," << logName << '\n' <<
				"AI LEVEL:," << +settings.aiLevel << "\n\n" <<
				"============================= MOVE INFORMATION =============================\n" <<
				"Player,Old Position,New Position,Jump Position,Move Type,Move Duration (us)" << std::endl;
	}
}

void GameLogger::logRedMove(Move move)
{
	if(file.is_open())
	{
		file << "Red," << +move.oldPos << "," << +move.newPos << "," << +move.jumpPos << "," << +move.moveType << std::endl;
	}
}

void GameLogger::logBlackMove(Move move, long long moveDurationMicro)
{
	if(file.is_open())
	{
		file << "Black," << +move.oldPos << "," << +move.newPos << "," << +move.jumpPos << "," << +move.moveType << "," << moveDurationMicro << std::endl;
	}
}

void GameLogger::logGameOver(GameState gameState)
{
	if(file.is_open())
	{
		file << "\nGame Over State:," << +gameState;

		if(gameState == GAME_OVER_BLACK_WIN)
			file << ",(Black Win)";
		else if(gameState == GAME_OVER_RED_WIN)
			file << ",(Red Win)";
		else if(gameState == GAME_OVER_TIE)
			file << ",(Tie)";

		file.close();
	}
}
