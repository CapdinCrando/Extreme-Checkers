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

void GameEngine::move(Move move)
{
	gameBoard.move(move.oldPos, move.newPos);
}


SquareState GameEngine::getSquareState(boardpos_t index)
{
	return gameBoard.getSquareState(index);
}

// TODO: Add jumps & multijumps
std::vector<Move> GameEngine::getPossibleMoves(boardpos_t pos)
{
	std::vector<Move> testMoves;
	Move m;
	m.oldPos = pos;
	for(uint8_t i = 0; i < 4; i++)
	{
		// Get move
		boardpos_t move = cornerList[pos][i];

		// Check if position is invalid
		if(move != BOARD_POS_INVALID)
		{
			// Check if space is empty
			if(gameBoard.getSquareState(move) == SQUARE_EMPTY)
			{
				// Add move to potential moves
				m.newPos = move;
				m.moveType = MOVE_JUMP;
				testMoves.push_back(m);
			}
		}
	}
	return testMoves;
}
