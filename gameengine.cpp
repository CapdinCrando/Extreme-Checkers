#include "gameengine.h"

#include <unordered_set>

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

// TODO: Add multijumps
std::vector<Move> GameEngine::getPossibleMoves(boardpos_t pos)
{
	std::vector<Move> testMoves;
	Move m;
	m.oldPos = pos;
	SquareState checkerState = getSquareState(pos);
	uint8_t cornerMax = 2;
	if(SQUARE_ISKING(checkerState)) cornerMax = 4;
	for(uint8_t i = 0; i < cornerMax; i++)
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
				m.moveType = MOVE_MOVE;
				testMoves.push_back(m);
			}
			else if(SQUARE_ISBLACK(move))
			{
				// Get jump
				boardpos_t jump = cornerList[move][i];
				// Check if position is invalid
				if(jump != BOARD_POS_INVALID)
				{
					// Check if space is empty
					if(gameBoard.getSquareState(jump) == SQUARE_EMPTY)
					{
						// Add move to potential moves
						m.newPos = jump;
						m.moveType = MOVE_JUMP;
						testMoves.push_back(m);
					}
				}
			}
		}
	}
	return testMoves;
}

// TODO: CHECK FOR MOVE
Move GameEngine::getAIMove()
{
	Move m;
	std::vector<Move> moves;
	for(uint8_t i = 0; i < SQUARE_COUNT; i++)
	{
		SquareState state = this->getSquareState(i);
		if(SQUARE_ISBLACK(state))
		{
			uint8_t cornerMin = 2;
			if(SQUARE_ISKING(state)) cornerMin = 0;
			for(uint8_t j = cornerMin; j < 4; j++)
			{
				// Get move
				boardpos_t move = cornerList[i][j];

				// Check if position is invalid
				if(move != BOARD_POS_INVALID)
				{
					// Check if space is empty
					if(gameBoard.getSquareState(move) == SQUARE_EMPTY)
					{
						// Add move to potential moves
						m.oldPos = i;
						m.newPos = move;
						m.moveType = MOVE_MOVE;
						moves.push_back(m);
					}
				}
			}
		}
	}
	return moves[rand() % moves.size()];
}
