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
	bitboard_t mask1 = (1 << pos1);
	bitboard_t mask2 = (1 << pos2);

	boardState.isOccupiedBoard |= mask2;
	boardState.isOccupiedBoard &= ~mask1;

	if(boardState.isBlackBoard & mask1) boardState.isBlackBoard |= mask2;
	else boardState.isBlackBoard &= ~mask2;

	if(boardState.isKingBoard & mask1) boardState.isKingBoard |= mask2;
	else boardState.isKingBoard &= ~mask2;
}

void GameBoard::setSquareState(boardpos_t index, SquareState state)
{
	bitboard_t mask = (1 << index);

	if(SQUARE_ISNOTEMPTY(state)) boardState.isOccupiedBoard |= mask;
	else boardState.isOccupiedBoard &= ~mask;

	if(SQUARE_ISBLACK(state)) boardState.isBlackBoard |= mask;
	else boardState.isBlackBoard &= ~mask;

	if(SQUARE_ISKING(state)) boardState.isKingBoard |= mask;
	else boardState.isKingBoard &= ~mask;
}


void GameBoard::makeEmpty(boardpos_t index)
{
	boardState.isOccupiedBoard &= ~(1 << index);
}

bool GameBoard::isEmpty(boardpos_t index)
{
	return (boardState.isOccupiedBoard & (1 << index)) == 0;
}

bool GameBoard::isOccupied(boardpos_t index)
{
	return boardState.isOccupiedBoard & (1 << index);
}

bool GameBoard::isRed(boardpos_t index)
{
	return (boardState.isBlackBoard & (1 << index)) == 0;
}

bool GameBoard::isBlack(boardpos_t index)
{
	return boardState.isBlackBoard & (1 << index);
}

bool GameBoard::isKing(boardpos_t index)
{
	return boardState.isKingBoard & (1 << index);
}

SquareState GameBoard::getSquareState(boardpos_t index)
{
	boardstate_t state = SQUARE_EMPTY;

	bitboard_t mask = (1 << index);
	if(boardState.isOccupiedBoard & mask)
	{
		state |= BIT_ISEMPTY;
		if(boardState.isBlackBoard & mask) state |= BIT_ISBLACK;
		if(boardState.isKingBoard & mask) state |= BIT_ISKING;
	}

	return static_cast<SquareState>(state);
}

bool GameBoard::kingPiece(boardpos_t pos)
{
	boardState.isKingBoard |= (1 << pos);
	return boardState.isOccupiedBoard & pos;
}
