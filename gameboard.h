#ifndef GAMEBOARD_H
#define GAMEBOARD_H

#include <cinttypes>
#include <bitset>

#define SQUARE_COUNT 32
#define BITS_PER_SQUARE 3

#define SQUARE_ISKING(state) state & 0x1
#define SQUARE_ISBLACK(state) state & 0x2
#define SQUARE_ISEMPTY(state) (state & 0x4) == 0
#define SQUARE_ISNOTEMPTY(state) state & 0x4

enum SquareState : uint8_t {
	SQUARE_EMPTY = 0,
	SQUARE_RED = 4,
	SQUARE_RED_KING = 5,
	SQUARE_BLACK = 6,
	SQUARE_BLACK_KING = 7
};

typedef int8_t boardpos_t;

struct Move {
	boardpos_t oldPos = -1;
	boardpos_t newPos = -1;
	boardpos_t jumpPos = -1;
};

typedef std::bitset<SQUARE_COUNT*BITS_PER_SQUARE> BoardState;

static const SquareState initialGame[SQUARE_COUNT] = {SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK,
											SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK,
											SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK,
											SQUARE_EMPTY, SQUARE_EMPTY, SQUARE_EMPTY, SQUARE_EMPTY,
											SQUARE_EMPTY, SQUARE_EMPTY, SQUARE_EMPTY, SQUARE_EMPTY,
											SQUARE_RED, SQUARE_RED, SQUARE_RED, SQUARE_RED,
											SQUARE_RED, SQUARE_RED, SQUARE_RED, SQUARE_RED,
											SQUARE_RED, SQUARE_RED, SQUARE_RED, SQUARE_RED };

class GameBoard
{
public:
	GameBoard();
	~GameBoard();

	void move(boardpos_t pos1, boardpos_t pos2);
	void setSquareState(boardpos_t index, SquareState state);
	SquareState getSquareState(boardpos_t index);
	void kingPiece(boardpos_t pos);

private:
	BoardState boardState;
};

#endif // GAMEBOARD_H
