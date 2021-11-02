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

typedef signed char result_t;

typedef unsigned char movetype_t;
enum MoveTypes : movetype_t
{
	MOVE_INVALID = 0,
	MOVE_MOVE = 1,
	MOVE_JUMP = 2,
	MOVE_JUMP_MULTI = 3,
};

struct Move
{
	unsigned char newPos : 5;
	unsigned char oldPos : 5;
	unsigned char jumpPos : 5;
	movetype_t moveType : 2;
	unsigned char isBlack : 1;
};

#endif // DEFINES_H
