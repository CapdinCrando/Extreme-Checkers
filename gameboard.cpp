#include "gameboard.h"

#include <iostream>

GameBoard::GameBoard()
{

}

GameBoard::~GameBoard()
{

}

/*BoardState* GameBoard::getBoardState()
{
	return &boardState;
}*/

void GameBoard::move(boardpos_t pos1, boardpos_t pos2)
{
	bitboard_t mask1 = (1 << pos1);
	bitboard_t mask2 = (1 << pos2);

	isOccupiedBoard |= mask2;
	isOccupiedBoard &= ~mask1;

	if(isBlackBoard & mask1) isBlackBoard |= mask2;
	else isBlackBoard &= ~mask2;

	if(isKingBoard & mask1) isKingBoard |= mask2;
	else isKingBoard &= ~mask2;
}

void GameBoard::setSquareState(boardpos_t index, SquareState state)
{
	bitboard_t mask = (1 << index);

	if(SQUARE_ISNOTEMPTY(state)) isOccupiedBoard |= mask;
	else isOccupiedBoard &= ~mask;

	if(SQUARE_ISBLACK(state)) isBlackBoard |= mask;
	else isBlackBoard &= ~mask;

	if(SQUARE_ISKING(state)) isKingBoard |= mask;
	else isKingBoard &= ~mask;
}


void GameBoard::makeEmpty(boardpos_t index)
{
	isOccupiedBoard &= ~(1 << index);
}

bool GameBoard::isEmpty(boardpos_t index)
{
	return (isOccupiedBoard & (1 << index)) == 0;
}

bool GameBoard::isOccupied(boardpos_t index)
{
	return isOccupiedBoard & (1 << index);
}

bool GameBoard::isRed(boardpos_t index)
{
	return (isBlackBoard & (1 << index)) == 0;
}

bool GameBoard::isBlack(boardpos_t index)
{
	return isBlackBoard & (1 << index);
}

bool GameBoard::isKing(boardpos_t index)
{
	return isKingBoard & (1 << index);
}

SquareState GameBoard::getSquareState(boardpos_t index)
{
	boardstate_t state = SQUARE_EMPTY;

	bitboard_t mask = (1 << index);
	if(isOccupiedBoard & mask)
	{
		state |= BIT_ISEMPTY;
		if(isBlackBoard & mask) state |= BIT_ISBLACK;
		if(isKingBoard & mask) state |= BIT_ISKING;
	}

	return static_cast<SquareState>(state);
}

bool GameBoard::kingPiece(boardpos_t pos)
{
	isKingBoard |= (1 << pos);
	return isOccupiedBoard & pos;
}
