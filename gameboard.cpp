#include "gameboard.h"

GameBoard::GameBoard()
{

}

GameBoard::~GameBoard()
{

}

void GameBoard::move(boardpos_t pos1, boardpos_t pos2)
{
	boardpos_t offset1 = pos1*BITS_PER_SQUARE;
	boardpos_t offset2 = pos2*BITS_PER_SQUARE;
	boardState[offset2] = boardState[offset1];
	boardState[offset2+1] = boardState[offset1+1];
	boardState[offset2+2] = boardState[offset1+2];
	boardState[offset1] = 0;
	boardState[offset1+1] = 0;
	boardState[offset1+2] = 0;
}

void GameBoard::setSquareState(boardpos_t index, SquareState state)
{
	boardpos_t offset = index*BITS_PER_SQUARE;
	boardState[offset] = state & (1 << 0);
	boardState[offset+1] = state & (1 << 1);
	boardState[offset+2] = state & (1 << 2);
}

SquareState GameBoard::getSquareState(boardpos_t index)
{
	boardpos_t offset = index*BITS_PER_SQUARE;
	return static_cast<SquareState>((boardState[offset]) | (boardState[offset+1] << 1) | (boardState[offset+2] << 2));
}
