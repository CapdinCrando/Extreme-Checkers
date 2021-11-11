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
	// Close file if open
	if(file.is_open()) file.close();

	if(settings.enableLogging)
	{
		std::string s;
		std::time_t rawTime = std::time(nullptr);
		std::tm localTime;
		localtime_s(&localTime, &rawTime);

		char logName[] = "log_YYYY-MM-DDT-HH-MM-SSZ.csv";
		std::strftime(&logName[4], sizeof(logName), "%FT%H-%M-%S", &localTime);

		file.open(logName, std::ios::out);
	}
}

void GameLogger::logMessage(std::string message)
{
	if(file.is_open()) file << message << std::endl;
}
