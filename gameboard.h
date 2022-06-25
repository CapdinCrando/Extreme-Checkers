#ifndef GAMEBOARD_H
#define GAMEBOARD_H

#include <cinttypes>

#define SQUARE_COUNT 32
#define BITS_PER_SQUARE 4

#define BIT_ISKING	0x1
#define BIT_ISBLACK 0x2
#define BIT_ISEMPTY 0x4

#define SQUARE_ISKING(state) state & BIT_ISKING
#define SQUARE_ISBLACK(state) state & BIT_ISBLACK
#define SQUARE_ISEMPTY(state) (state & BIT_ISEMPTY) == 0
#define SQUARE_ISNOTEMPTY(state) state & BIT_ISEMPTY

#define MOVE_ISJUMP(move) move.moveType >= MOVE_JUMP
#define MOVE_ISINVALID(move) move.moveType == MOVE_INVALID

#define BOARD_POS_INVALID -1

typedef uint8_t boardstate_t;
enum SquareState : boardstate_t {
	SQUARE_EMPTY = 0,
	SQUARE_RED = 4,
	SQUARE_RED_KING = 5,
	SQUARE_BLACK = 6,
	SQUARE_BLACK_KING = 7
};

typedef int8_t boardpos_t;

typedef uint32_t bitboard_t;
typedef uint32_t bitpos_t;

struct BoardState
{
	bitboard_t isOccupiedBoard, isBlackBoard, isKingBoard;
};

static const SquareState initialGame[SQUARE_COUNT] = {SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK,
											SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK,
											SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK, SQUARE_BLACK,
											SQUARE_EMPTY, SQUARE_EMPTY, SQUARE_EMPTY, SQUARE_EMPTY,
											SQUARE_EMPTY, SQUARE_EMPTY, SQUARE_EMPTY, SQUARE_EMPTY,
											SQUARE_RED, SQUARE_RED, SQUARE_RED, SQUARE_RED,
											SQUARE_RED, SQUARE_RED, SQUARE_RED, SQUARE_RED,
											SQUARE_RED, SQUARE_RED, SQUARE_RED, SQUARE_RED };

const boardpos_t cornerList[SQUARE_COUNT][4] = {
	{-1, -1, 4, 5},{-1, -1, 5, 6},{-1, -1, 6, 7},{-1, -1, 7, -1},
	{-1, 0, -1, 8},{0, 1, 8, 9},{1, 2, 9, 10},{2, 3, 10, 11},
	{4, 5, 12, 13},{5, 6, 13, 14},{6, 7, 14, 15},{7, -1, 15, -1},
	{-1, 8, -1, 16},{8, 9, 16, 17},{9, 10, 17, 18},{10, 11, 18, 19},
	{12, 13, 20, 21},{13, 14, 21, 22},{14, 15, 22, 23},{15, -1, 23, -1},
	{-1, 16, -1, 24},{16, 17, 24, 25},{17, 18, 25, 26},{18, 19, 26, 27},
	{20, 21, 28, 29},{21, 22, 29, 30},{22, 23, 30, 31},{23, -1, 31, -1},
	{-1, 24, -1, -1},{24, 25, -1, -1},{25, 26, -1, -1},{26, 27, -1, -1}
};

class GameBoard
{
public:
	GameBoard();
	~GameBoard();

	BoardState getBoardState();
	void move(boardpos_t pos1, boardpos_t pos2);
	void setSquareState(boardpos_t index, SquareState state);
	SquareState getSquareState(boardpos_t index);
	void makeEmpty(boardpos_t index);
	bool isOccupied(boardpos_t index);
	bool isEmpty(boardpos_t index);
	bool isRed(boardpos_t index);
	bool isBlack(boardpos_t index);
	bool isKing(boardpos_t index);
	bool kingPiece(boardpos_t pos);

private:
	BoardState boardState;
};

#endif // GAMEBOARD_H
