#include "gameboard.h"

#include <iostream>

GameBoard::GameBoard()
{

}

GameBoard::~GameBoard()
{

}

BoardState* GameBoard::getBoardState()
{
	return &boardState;
}

void GameBoard::move(boardpos_t pos1, boardpos_t pos2)
{
	boardState[pos2] = boardState[pos1];
	boardState[pos1] = SQUARE_EMPTY;
}

void GameBoard::setSquareState(boardpos_t index, SquareState state)
{
	boardState[index] = state;
}

SquareState GameBoard::getSquareState(boardpos_t index)
{
	return static_cast<SquareState>(boardState[index]);
}

bool GameBoard::kingPiece(boardpos_t pos)
{
	if(SQUARE_ISNOTEMPTY(boardState[pos]))
	{
		boardState[pos] |= 0x1;
		return true;
	}
	return false;
}
