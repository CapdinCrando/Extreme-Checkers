#ifndef DEFINES_H
#define DEFINES_H

#include <cinttypes>
#include <QStringList>
#include <vector>

enum GameState : int8_t
{
	GAME_ERROR = -1,
	GAME_RUNNING = 0,
	GAME_OVER_RED_WIN = 1,
	GAME_OVER_BLACK_WIN = 2,
	GAME_OVER_TIE = 3,
};

struct GameSettings
{
	unsigned char aiLevel = 0;
};

const QStringList aiModes = {	"Level 1 - Random AI",
								"Level 2 - Minimax AI",
								"Level 3 - Parallel AI",
								"Level 4 - Accelerated AI" };

#endif // DEFINES_H
