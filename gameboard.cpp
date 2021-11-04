#include "gameboard.h"

#include <iostream>

GameBoard::GameBoard()
{

}

GameBoard::~GameBoard()
{

}

BoardState GameBoard::getBoardState()
{
	return boardState;
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

bool GameBoard::kingPiece(boardpos_t pos)
{
	boardpos_t offset = pos*BITS_PER_SQUARE;
	if(boardState[offset + 2] != 0)
	{
		boardState[offset] = 1;
		return true;
	}
	return false;
}
