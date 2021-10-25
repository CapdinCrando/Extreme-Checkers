#include "gameengine.h"

#include <iostream>

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

// TODO: CHECK WIN STATE
void GameEngine::executeRedMove(Move move)
{
	std::cout << "Executing red move: " << +move.oldPos << "," << +move.jumpPos << "," << +move.newPos << std::endl;
	this->move(move);
	emit displayMove(move);
	if(move.jumpPos != BOARD_POS_INVALID)
	{
		gameBoard.setSquareState(move.jumpPos, SQUARE_EMPTY);
		// Check if can jump again
		std::vector<Move> newJumps;
		for(uint8_t i = 0; i < 4; i++)
		{
			// Get move
			boardpos_t cornerPiece = cornerList[move.newPos][i];

			// Check if position is invalid
			if(cornerPiece != BOARD_POS_INVALID)
			{
				SquareState cornerState = gameBoard.getSquareState(cornerPiece);
				if(SQUARE_ISNOTEMPTY(cornerState))
				{
					if(SQUARE_ISBLACK(cornerState))
					{
						boardpos_t jump = cornerList[cornerPiece][i];
						if(jump != BOARD_POS_INVALID)
						{
							if(SQUARE_ISEMPTY(gameBoard.getSquareState(jump)))
							{
								Move newJump;
								newJump.oldPos = move.newPos;
								newJump.newPos = jump;
								newJump.jumpPos = cornerPiece;
								newJumps.push_back(newJump);
							}
						}
					}
				}
			}
		}
		if(!newJumps.empty())
		{
			emit displayMultiJump(newJumps, gameBoard.getSquareState(move.newPos));
			return;
		}
	}
	executeBlackMove(getAIMove());
}

// TODO: CHECK WIN STATE
void GameEngine::executeBlackMove(Move move)
{
	std::cout << "Executing black move: " << +move.oldPos << "," << +move.jumpPos << "," << +move.newPos << std::endl;
	this->move(move);
	emit displayMove(move);
	if(move.jumpPos != BOARD_POS_INVALID)
	{
		gameBoard.setSquareState(move.jumpPos, SQUARE_EMPTY);
		// Check if can jump again
		std::vector<Move> newJumps;
		for(uint8_t i = 0; i < 4; i++)
		{
			// Get move
			boardpos_t cornerPiece = cornerList[move.newPos][i];

			// Check if position is invalid
			if(cornerPiece != BOARD_POS_INVALID)
			{
				SquareState cornerState = gameBoard.getSquareState(cornerPiece);
				if(SQUARE_ISNOTEMPTY(cornerState))
				{
					if(!(SQUARE_ISBLACK(cornerState)))
					{
						boardpos_t jump = cornerList[cornerPiece][i];
						if(jump != BOARD_POS_INVALID)
						{
							if(SQUARE_ISEMPTY(gameBoard.getSquareState(jump)))
							{
								Move newJump;
								newJump.oldPos = move.newPos;
								newJump.newPos = jump;
								newJump.jumpPos = cornerPiece;
								newJumps.push_back(newJump);
							}
						}
					}
				}
			}
		}
		if(newJumps.empty())
		{
			emit blackMoveFinished();
		}
		else
		{
			executeBlackMove(newJumps[rand() % newJumps.size()]);
		}
	}
	else
	{
		emit blackMoveFinished();
	}
}

SquareState GameEngine::getSquareState(boardpos_t index)
{
	return gameBoard.getSquareState(index);
}

std::vector<Move> GameEngine::getPossibleMoves(boardpos_t pos)
{
	std::vector<Move> testMoves;
	std::vector<Move> testJumps;
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
			SquareState moveState = gameBoard.getSquareState(move);
			if(SQUARE_ISEMPTY(moveState))
			{
				// Add move to potential moves
				m.newPos = move;
				m.jumpPos = -1;
				testMoves.push_back(m);
			}
			else if(SQUARE_ISBLACK(moveState))
			{
				// Get jump
				boardpos_t jump = cornerList[move][i];
				// Check if position is invalid
				if(jump != BOARD_POS_INVALID)
				{
					// Check if space is empty
					if(SQUARE_ISEMPTY(gameBoard.getSquareState(jump)))
					{
						// Add move to potential moves
						m.newPos = jump;
						m.jumpPos = move;
						testJumps.push_back(m);
					}
				}
			}
		}
	}
	if(testJumps.empty()) return testMoves;
	else return testJumps;
}

Move GameEngine::getAIMove()
{
	Move m;
	std::vector<Move> moves;
	std::vector<Move> jumps;
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
					SquareState moveState = gameBoard.getSquareState(move);
					if(SQUARE_ISEMPTY(moveState))
					{
						// Add move to potential moves
						m.oldPos = i;
						m.newPos = move;
						m.jumpPos = -1;
						moves.push_back(m);
					}
					else if(!(SQUARE_ISBLACK(moveState)))
					{
						// Get jump
						boardpos_t jump = cornerList[move][j];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(SQUARE_ISEMPTY(gameBoard.getSquareState(jump)))
							{
								// Add move to potential moves
								m.oldPos = i;
								m.newPos = jump;
								m.jumpPos = move;
								jumps.push_back(m);
							}
						}
					}
				}
			}
		}
	}
	if(jumps.empty()) return moves[rand() % moves.size()];
	else return jumps[rand() % jumps.size()];
}
