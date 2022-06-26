#include "gameengine.h"
#include "ai/gpuutility.h"

#include <chrono>

#ifdef QT_DEBUG
#define PROFILING
#endif

#ifdef PROFILING
#include <iostream>
#endif

GameEngine::GameEngine()
{
	aiManager = new AIManager();
	logger = new GameLogger();

	connect(this, &GameEngine::executeBlackMove, this, &GameEngine::calculateMove, Qt::QueuedConnection);
	connect(this, &GameEngine::logRedMove, logger, &GameLogger::logRedMove, Qt::QueuedConnection);
	connect(this, &GameEngine::logBlackMove, logger, &GameLogger::logBlackMove, Qt::QueuedConnection);
	connect(this, &GameEngine::gameOver, logger, &GameLogger::logGameOver, Qt::QueuedConnection);
}

GameEngine::~GameEngine()
{
	delete aiManager;
	delete logger;
}

void GameEngine::resetGame()
{
	for(boardpos_t i = 0; i < SQUARE_COUNT; i++)
	{
		gameBoard.setSquareState(i, initialGame[i]);
	}
	jumpExists = false;
	logger->openLogFile(settings);
}

void GameEngine::saveSettings(GameSettings settings)
{
	aiManager->selectAI(settings.aiLevel);
	this->settings = settings;

#ifdef QT_DEBUG
	std::cout << "Setting: " << +settings.aiLevel << std::endl;
#endif
}

void GameEngine::move(Move move)
{
	gameBoard.move(move.oldPos, move.newPos);
}

void GameEngine::executeRedMove(Move move)
{
#ifdef QT_DEBUG
	std::cout << "Executing red move: " << +move.oldPos << "," << +move.jumpPos << "," << +move.newPos << std::endl;
#endif
	this->move(move);

	// Check for king
	if(move.newPos < 4)
	{
		emit displayMove(move, gameBoard.kingPiece(move.newPos));
	}
	else emit displayMove(move, false);
	emit logRedMove(move);

	if(MOVE_ISJUMP(move))
	{
		gameBoard.makeEmpty(move.jumpPos);
		// Check if can jump again
		std::vector<Move> newJumps;
		for(uint8_t i = 0; i < 4; i++)
		{
			// Get move
			boardpos_t cornerPiece = cornerList[move.newPos][i];

			// Check if position is invalid
			if(cornerPiece != BOARD_POS_INVALID)
			{
				if(gameBoard.isOccupiedBlack(cornerPiece))
				{
					boardpos_t jump = cornerList[cornerPiece][i];
					if(jump != BOARD_POS_INVALID)
					{
						if(gameBoard.isEmpty(jump))
						{
							Move newJump;
							newJump.oldPos = move.newPos;
							newJump.newPos = jump;
							newJump.jumpPos = cornerPiece;
							newJump.moveType = MOVE_JUMP;
							newJumps.push_back(newJump);
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
	if(checkGameOver(false)) return;
	emit executeBlackMove();
}

void GameEngine::calculateMove()
{
	auto start = std::chrono::high_resolution_clock::now();
	Move move;
	try
	{
		move = aiManager->getMove(gameBoard);
	}
	catch (const std::exception& e)
	{
		emit printError(e);
		return;
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();


	emit logBlackMove(move, duration);

#ifdef PROFILING
	std::cout << "Black move calculated in " << duration << " us" << std::endl;
#endif

	long long remainingWait = (CALCULATE_TIME_US - duration)/1000;
	//std::cout << "Remaining time: " << remainingWait << std::endl;
	if(remainingWait > 0)
	{
		QTimer::singleShot(remainingWait, [=]()
		{
			handleBlackMove(move);
		});
	}
	else handleBlackMove(move);
}

void GameEngine::handleBlackMove(Move move)
{
	if(MOVE_ISINVALID(move)) return;

#ifdef QT_DEBUG
	std::cout << "Executing black move: " << +move.oldPos << "," << +move.jumpPos << "," << +move.newPos << std::endl;
#endif
	this->move(move);

	// Check for king
	if(move.newPos > 27)
	{
		emit displayMove(move, gameBoard.kingPiece(move.newPos));
	}
	else emit displayMove(move, false);

	if(MOVE_ISJUMP(move))
	{
		gameBoard.makeEmpty(move.jumpPos);
		if(move.moveType == MOVE_JUMP_MULTI)
		{
			emit executeBlackMove();
		}
		else
		{
			emit blackMoveFinished();
			if(checkGameOver(true)) return;
		}
	}
	else
	{
		emit blackMoveFinished();
		if(checkGameOver(true)) return;
	}
}

bool GameEngine::checkGameOver(bool isBlackTurn)
{
	result_t blackCount = 0;
	result_t redCount = 0;
	bool redMoveFound = false;
	bool blackMoveFound = false;
	jumpExists = false;
	for(boardpos_t pos = 0; pos < SQUARE_COUNT; pos++)
	{
		if(gameBoard.isOccupied(pos))
		{
			if(gameBoard.isBlack(pos))
			{
				blackCount += 1;
				uint8_t cornerMin = 2;
				if(gameBoard.isKing(pos)) cornerMin = 0;
				for(uint8_t i = cornerMin; i < 4; i++)
				{
					// Get move
					boardpos_t move = cornerList[pos][i];

					// Check if position is invalid
					if(move != BOARD_POS_INVALID)
					{
						// Check if space is empty
						if(gameBoard.isEmpty(move))
						{
							blackMoveFound = true;
						}
						else if(gameBoard.isRed(move))
						{
							// Get jump
							boardpos_t jump = cornerList[move][i];
							// Check if position is invalid
							if(jump != BOARD_POS_INVALID)
							{
								// Check if space is empty
								if(gameBoard.isEmpty(jump))
								{
									// Add jump to potential moves
									blackMoveFound = true;
								}
							}
						}
					}
				}
			}
			else
			{
				redCount += 1;
				uint8_t cornerMax = 2;
				if(gameBoard.isKing(pos)) cornerMax = 4;
				for(uint8_t i = 0; i < cornerMax; i++)
				{
					// Get move
					boardpos_t move = cornerList[pos][i];

					// Check if position is invalid
					if(move != BOARD_POS_INVALID)
					{
						// Check if space is empty
						if(gameBoard.isEmpty(move))
						{
							redMoveFound = true;
						}
						else if(gameBoard.isBlack(move))
						{
							// Get jump
							boardpos_t jump = cornerList[move][i];
							// Check if position is invalid
							if(jump != BOARD_POS_INVALID)
							{
								// Check if space is empty
								if(gameBoard.isEmpty(jump))
								{
									// Add jump to potential moves
									redMoveFound = true;
									jumpExists = true;
								}
							}
						}
					}
				}
			}
		}
	}
	if(blackCount == 0)
	{
		if(redCount != 0)
		{
			// Red win
			emit gameOver(GAME_OVER_RED_WIN);
			return true;
		}
	}
	else if(redCount == 0)
	{
		if(blackCount != 0)
		{
			// Black win
			emit gameOver(GAME_OVER_BLACK_WIN);
			return true;
		}
	}
	else
	{
		if(!blackMoveFound)
		{
			if(redMoveFound)
			{
				// RED WIN
				if(!isBlackTurn)
				{
					emit gameOver(GAME_OVER_RED_WIN);
					return true;
				}
			}
			else
			{
				// TIE
				emit gameOver(GAME_OVER_TIE);
				return true;
			}
		}
		else if(!redMoveFound)
		{
			if(blackMoveFound)
			{
				// BLACK WIN
				if(isBlackTurn)
				{
					emit gameOver(GAME_OVER_BLACK_WIN);
					return true;
				}
			}
			else
			{
				// TIE
				emit gameOver(GAME_OVER_TIE);
				return true;
			}
		}
	}
	return false;
}

SquareState GameEngine::getSquareState(boardpos_t index)
{
	return gameBoard.getSquareState(index);
}

std::vector<Move> GameEngine::getRedMoves(boardpos_t pos)
{
	Move m;
	std::vector<Move> moves;
	std::vector<Move> jumps;
	if(gameBoard.isRed(pos))
	{
		uint8_t cornerMax = 2;
		if(gameBoard.isKing(pos)) cornerMax = 4;
		for(uint8_t j = 0; j < cornerMax; j++)
		{
			// Get move
			boardpos_t move = cornerList[pos][j];
			// Check if position is invalid
			if(move != BOARD_POS_INVALID)
			{
				// Check if space is empty
				if(jumpExists)
				{
					if(gameBoard.isOccupiedBlack(move))
					{
						// Get jump
						boardpos_t jump = cornerList[move][j];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(gameBoard.isEmpty(jump))
							{
								// Add move to potential moves
								m.oldPos = pos;
								m.newPos = jump;
								m.jumpPos = move;
								// Check for multi
								m.moveType = MOVE_JUMP;
								for(uint8_t k = 0; k < 4; k++)
								{
									boardpos_t moveMulti = cornerList[jump][k];
									// Check if position is invalid
									if(moveMulti != BOARD_POS_INVALID)
									{
										if(moveMulti != move)
										{
											if(gameBoard.isOccupiedBlack(moveMulti))
											{
												boardpos_t jumpMulti = cornerList[moveMulti][k];
												if(jumpMulti != BOARD_POS_INVALID)
												{
													if(gameBoard.isEmpty(jumpMulti))
													{
														m.moveType = MOVE_JUMP_MULTI;
														break;
													}
												}

											}
										}
									}
								}
								jumps.push_back(m);
							}
						}
					}
				}
				else if(gameBoard.isEmpty(move))
				{
					// Add move to potential moves
					m.oldPos = pos;
					m.newPos = move;
					m.moveType = MOVE_MOVE;
					moves.push_back(m);
				}
			}
		}
	}
	if(jumps.empty()) return moves;
	return jumps;
}

bool GameEngine::canBlackMove()
{
	for(uint8_t pos = 0; pos < SQUARE_COUNT; pos++)
	{
		if(gameBoard.isOccupiedBlack(pos))
		{
			uint8_t cornerMin = 2;
			if(gameBoard.isKing(pos)) cornerMin = 0;
			for(uint8_t j = cornerMin; j < 4; j++)
			{
				// Get move
				boardpos_t move = cornerList[pos][j];
				// Check if position is invalid
				if(move != BOARD_POS_INVALID)
				{
					// Check if space is empty
					if(gameBoard.isEmpty(move))
					{
						// Add move to potential moves
						return true;
					}
					else if(gameBoard.isRed(move))
					{
						// Get jump
						boardpos_t jump = cornerList[move][j];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(gameBoard.isEmpty(jump))
							{
								// Add move to potential moves
								return true;
							}
						}
					}
				}
			}
		}
	}
	return false;
}

Move GameEngine::getAIMove()
{
	Move move = aiManager->getMove(gameBoard);
	if(MOVE_ISINVALID(move))
	{
		return Move();
	}
	return move;
}
