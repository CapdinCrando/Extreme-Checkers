#include "gameengine.h"

GameEngine::GameEngine()
{

}

GameEngine::~GameEngine()
{

}

void GameEngine::resetGame()
{
	for(boardpos_t i = 0; i < SQUARE_COUNT; i++)
	{
		gameBoard.setSquareState(i, initialGame[i]);
	}
}

void GameEngine::move(boardpos_t pos1, boardpos_t pos2)
{
	gameBoard.move(pos1, pos2);
}


SquareState GameEngine::getSquareState(boardpos_t index)
{
	return gameBoard.getSquareState(index);
}
