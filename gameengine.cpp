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
	jumpExists = false;
}

void GameEngine::saveSettings(GameSettings settings)
{
	this->settings = settings;
	std::cout << "Setting: " << +settings.aiLevel << std::endl;
}

void GameEngine::move(Move move)
{
	gameBoard.move(move.oldPos, move.newPos);
}

void GameEngine::executeRedMove(Move move)
{
	std::cout << "Executing red move: " << +move.oldPos << "," << +move.jumpPos << "," << +move.newPos << std::endl;
	this->move(move);
	emit displayMove(move, this->checkForKing(move.newPos));
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
	if(checkRedWin())
	{
		emit gameOver(GAME_OVER_RED_WIN);
		return;
	}
	executeBlackMove();
}

void GameEngine::executeBlackMove()
{
	Move move = this->getAIMove();
	if(move.oldPos == -1) return;

	std::cout << "Executing black move: " << +move.oldPos << "," << +move.jumpPos << "," << +move.newPos << std::endl;
	this->move(move);
	emit displayMove(move, this->checkForKing(move.newPos));
	if(move.jumpPos != BOARD_POS_INVALID)
	{
		gameBoard.setSquareState(move.jumpPos, SQUARE_EMPTY);
		// Check if can jump again
		bool canJumpAgain = false;
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
								canJumpAgain = true;
								break;
							}
						}
					}
				}
			}
		}
		if(canJumpAgain)
		{
			executeBlackMove();
		}
		else
		{
			emit blackMoveFinished();
			if(!checkIfRedMoveExists()) checkRedTie();
		}
	}
	else
	{
		emit blackMoveFinished();
		if(!checkIfRedMoveExists()) checkRedTie();
	}
	if(checkBlackWin())
	{
		emit gameOver(GAME_OVER_BLACK_WIN);
		return;
	}
}

bool GameEngine::checkForKing(boardpos_t pos)
{
	if((pos < 4) || (pos > 27))
	{
		gameBoard.kingPiece(pos);
		return true;
	}
	return false;
}

bool GameEngine::checkBlackWin()
{
	for(uint8_t i = 0; i < SQUARE_COUNT; i++)
	{
		SquareState state = gameBoard.getSquareState(i);
		if(SQUARE_ISNOTEMPTY(state))
		{
			if(!(SQUARE_ISBLACK(state)))
			{
				return false;
			}
		}
	}
	return true;
}

bool GameEngine::checkRedWin()
{
	for(uint8_t i = 0; i < SQUARE_COUNT; i++)
	{
		SquareState state = gameBoard.getSquareState(i);
		if(SQUARE_ISNOTEMPTY(state))
		{
			if(SQUARE_ISBLACK(state))
			{
				return false;
			}
		}
	}
	return true;
}

void GameEngine::checkRedTie()
{
	// TODO: Performance improvement
	if(getAllBlackMoves().empty())
	{
		emit gameOver(GAME_OVER_TIE);
	}
	else
	{
		emit gameOver(GAME_OVER_BLACK_WIN);
	}
}

void GameEngine::checkBlackTie()
{
	if(checkIfRedMoveExists())
	{
		emit gameOver(GAME_OVER_RED_WIN);
	}
	else
	{
		emit gameOver(GAME_OVER_TIE);
	}
}

SquareState GameEngine::getSquareState(boardpos_t index)
{
	return gameBoard.getSquareState(index);
}

bool GameEngine::checkIfRedMoveExists()
{
	jumpExists = false;
	bool moveExists = false;
	for(uint8_t i = 0; i < SQUARE_COUNT; i++)
	{
		SquareState state = this->getSquareState(i);
		if(SQUARE_ISNOTEMPTY(state))
		{
			if(!(SQUARE_ISBLACK(state)))
			{
				uint8_t cornerMax = 2;
				if(SQUARE_ISKING(state)) cornerMax = 4;
				for(uint8_t j = 0; j < cornerMax; j++)
				{
					// Get move
					boardpos_t move = cornerList[i][j];
					// Check if position is invalid
					if(move != BOARD_POS_INVALID)
					{
						// Check if space is empty
						SquareState moveState = gameBoard.getSquareState(move);
						if(SQUARE_ISNOTEMPTY(moveState))
						{
							if(SQUARE_ISBLACK(moveState))
							{
								// Get jump
								boardpos_t jump = cornerList[move][j];
								// Check if position is invalid
								if(jump != BOARD_POS_INVALID)
								{
									// Check if space is empty
									if(SQUARE_ISEMPTY(gameBoard.getSquareState(jump)))
									{
										jumpExists = true;
										return true;
									}
								}
							}
						}
						else moveExists = true;
					}
				}
			}
		}
	}
	return moveExists;
}

std::vector<Move> GameEngine::getRedMoves(boardpos_t pos)
{
	std::vector<Move> testMoves;
	std::vector<Move> testJumps;
	Move m;
	m.oldPos = pos;
	SquareState checkerState = getSquareState(pos);
	if(SQUARE_ISNOTEMPTY(checkerState))
	{
		if(!(SQUARE_ISBLACK(checkerState)))
		{
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
					if(jumpExists)
					{
						if(SQUARE_ISBLACK(moveState))
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
					else if(SQUARE_ISEMPTY(moveState))
					{
						// Add move to potential moves
						m.newPos = move;
						m.jumpPos = -1;
						testMoves.push_back(m);
					}
				}
			}
		}
	}

	if(testJumps.empty()) return testMoves;
	return testJumps;
}


std::vector<Move> GameEngine::getAllBlackMoves()
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
	if(jumps.empty()) return moves;
	return jumps;
}

Move GameEngine::getAIMove()
{
	std::vector<Move> moves = getAllBlackMoves();
	if(moves.empty())
	{
		checkBlackTie();
		return Move();
	}
	return moves[rand() % moves.size()];
}
