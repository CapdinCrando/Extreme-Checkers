#ifndef DEFINES_H
#define DEFINES_H

#include <cinttypes>
#include <vector>

#define NODE_DEPTH_MINIMAX 7
#define NODE_DEPTH_PARALLEL 8
#define NODE_DEPTH_GPU 6

#define CALCULATE_TIME_US 500000

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
	bool enableLogging = 0;
};

typedef unsigned char depth_t;

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
};

typedef signed char result_t;
enum ResultType : result_t
{
	RESULT_TIE = 0,
	RESULT_RED_WIN = -24,
	RESULT_BLACK_WIN = 24
};

#endif // DEFINES_H
