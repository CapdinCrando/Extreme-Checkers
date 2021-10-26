#ifndef DEFINES_H
#define DEFINES_H

#include <cinttypes>

enum GameState : int8_t
{
	GAME_ERROR = -1,
	GAME_RUNNING = 0,
	GAME_OVER_RED_WIN = 1,
	GAME_OVER_BLACK_WIN = 2,
	GAME_OVER_TIE = 3,
};

#endif // DEFINES_H
