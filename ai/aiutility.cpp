#include "aiutility.h"
#include "defines.h"
#include <algorithm>

Table AIUtility::redTable = Table();
Table AIUtility::blackTable = Table();

std::vector<Move>* AIUtility::getAllBlackMoves(GameBoard &board)
{
	Move m;
	std::vector<Move> *moves = new std::vector<Move>;
	std::vector<Move> *jumps = new std::vector<Move>;
	for(uint8_t i = 0; i < SQUARE_COUNT; i++)
	{
		if(board.isOccupiedBlack(i))
		{
			uint8_t cornerMin = 2;
			if(board.isKing(i)) cornerMin = 0;
			for(uint8_t j = cornerMin; j < 4; j++)
			{
				// Get move
				boardpos_t move = cornerList[i][j];
				// Check if position is invalid
				if(move != BOARD_POS_INVALID)
				{
					// Check if space is empty
					if(board.isEmpty(move))
					{
						// Add move to potential moves
						m.oldPos = i;
						m.newPos = move;
						m.moveType = MOVE_MOVE;
						moves->push_back(m);
					}
					else if(board.isRed(move))
					{
						// Get jump
						boardpos_t jump = cornerList[move][j];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(board.isEmpty(jump))
							{
								// Add move to potential moves
								m.oldPos = i;
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
											if(board.isOccupiedRed(moveMulti))
											{
												boardpos_t jumpMulti = cornerList[moveMulti][k];
												if(jumpMulti != BOARD_POS_INVALID)
												{
													if(board.isEmpty(jumpMulti))
													{
														m.moveType = MOVE_JUMP_MULTI;
														break;
													}
												}
											}
										}
									}
								}
								jumps->push_back(m);
							}
						}
					}
				}
			}
		}
	}
	if(jumps->empty())
	{
		delete jumps;
		return moves;
	}
	delete moves;
	return jumps;
}

std::vector<Move>* AIUtility::getAllBlackJumps(GameBoard &board, boardpos_t pos)
{
	Move m;
	std::vector<Move> *jumps = new std::vector<Move>;
	if(board.isOccupied(pos))
	{
		for(uint8_t j = 0; j < 4; j++)
		{
			// Get move
			boardpos_t move = cornerList[pos][j];
			// Check if position is invalid
			if(move != BOARD_POS_INVALID)
			{
				// Check if space is empty
				if(board.isOccupiedRed(move))
				{
					// Get jump
					boardpos_t jump = cornerList[move][j];
					// Check if position is invalid
					if(jump != BOARD_POS_INVALID)
					{
						// Check if space is empty
						if(board.isEmpty(jump))
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
										if(board.isOccupiedRed(moveMulti))
										{
											boardpos_t jumpMulti = cornerList[moveMulti][k];
											if(jumpMulti != BOARD_POS_INVALID)
											{
												if(board.isEmpty(jumpMulti))
												{
													m.moveType = MOVE_JUMP_MULTI;
													break;
												}
											}
										}
									}
								}
							}
							jumps->push_back(m);
						}
					}
				}
			}
		}
	}
	return jumps;
}

std::vector<Move>* AIUtility::getAllRedMoves(GameBoard &board)
{
	Move m;
	std::vector<Move> *moves = new std::vector<Move>;
	std::vector<Move> *jumps = new std::vector<Move>;
	for(uint8_t i = 0; i < SQUARE_COUNT; i++)
	{
		if(board.isOccupiedRed(i))
		{
			uint8_t cornerMax = 2;
			if(board.isKing(i)) cornerMax = 4;
			for(uint8_t j = 0; j < cornerMax; j++)
			{
				// Get move
				boardpos_t move = cornerList[i][j];
				// Check if position is invalid
				if(move != BOARD_POS_INVALID)
				{
					// Check if space is empty
					if(board.isEmpty(move))
					{
						// Add move to potential moves
						m.oldPos = i;
						m.newPos = move;
						m.moveType = MOVE_MOVE;
						moves->push_back(m);
					}
					else if(board.isBlack(move))
					{
						// Get jump
						boardpos_t jump = cornerList[move][j];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(board.isEmpty(jump))
							{
								// Add move to potential moves
								m.oldPos = i;
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
											if(board.isOccupiedBlack(moveMulti))
											{
												boardpos_t jumpMulti = cornerList[moveMulti][k];
												if(jumpMulti != BOARD_POS_INVALID)
												{
													if(board.isEmpty(jumpMulti))
													{
														m.moveType = MOVE_JUMP_MULTI;
														break;
													}
												}
											}
										}
									}
								}
								jumps->push_back(m);
							}
						}
					}
				}
			}
		}
	}
	if(jumps->empty())
	{
		delete jumps;
		return moves;
	}
	delete moves;
	return jumps;
}

std::vector<Move>* AIUtility::getAllRedJumps(GameBoard &board, boardpos_t pos)
{
	Move m;
	std::vector<Move> *jumps = new std::vector<Move>;
	if(board.isOccupied(pos))
	{
		for(uint8_t j = 0; j < 4; j++)
		{
			// Get move
			boardpos_t move = cornerList[pos][j];
			// Check if position is invalid
			if(move != BOARD_POS_INVALID)
			{
				// Check if space is empty
				if(board.isOccupiedBlack(move))
				{
					// Get jump
					boardpos_t jump = cornerList[move][j];
					// Check if position is invalid
					if(jump != BOARD_POS_INVALID)
					{
						// Check if space is empty
						if(board.isEmpty(jump))
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
										if(board.isOccupiedBlack(moveMulti))
										{
											boardpos_t jumpMulti = cornerList[moveMulti][k];
											if(jumpMulti != BOARD_POS_INVALID)
											{
												if(board.isEmpty(jumpMulti))
												{
													m.moveType = MOVE_JUMP_MULTI;
													break;
												}
											}
										}
									}
								}
							}
							jumps->push_back(m);
						}
					}
				}
			}
		}
	}
	return jumps;
}

bool AIUtility::evalBoardResult(GameBoard &board, result_t& resultOut)
{
	result_t blackCount = 0;
	result_t redCount = 0;
	bool redMoveFound = false;
	bool blackMoveFound = false;
	for(boardpos_t pos = 0; pos < SQUARE_COUNT; pos++)
	{
		if(board.isOccupied(pos))
		{
			if(board.isBlack(pos))
			{
				blackCount += 1;
				uint8_t cornerMin = 2;
				if(board.isKing(pos)) cornerMin = 0;
				for(uint8_t i = cornerMin; i < 4; i++)
				{
					// Get move
					boardpos_t move = cornerList[pos][i];

					// Check if position is invalid
					if(move != BOARD_POS_INVALID)
					{
						// Check if space is empty
						if(board.isEmpty(move))
						{
							blackMoveFound = true;
						}
						else if(board.isRed(move))
						{
							// Get jump
							boardpos_t jump = cornerList[move][i];
							// Check if position is invalid
							if(jump != BOARD_POS_INVALID)
							{
								// Check if space is empty
								if(board.isEmpty(jump))
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
				if(board.isKing(pos)) cornerMax = 4;
				for(uint8_t i = 0; i < cornerMax; i++)
				{
					// Get move
					boardpos_t move = cornerList[pos][i];

					// Check if position is invalid
					if(move != BOARD_POS_INVALID)
					{
						// Check if space is empty
						if(board.isEmpty(move))
						{
							redMoveFound = true;
						}
						else if(board.isBlack(move))
						{
							// Get jump
							boardpos_t jump = cornerList[move][i];
							// Check if position is invalid
							if(jump != BOARD_POS_INVALID)
							{
								// Check if space is empty
								if(board.isEmpty(jump))
								{
									// Add jump to potential moves
									redMoveFound = true;
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
			resultOut = RESULT_RED_WIN;
			return true;
		}
	}
	else if(redCount == 0)
	{
		if(blackCount != 0)
		{
			// Black win
			resultOut = RESULT_BLACK_WIN;
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
				resultOut = RESULT_RED_WIN;
				return true;
			}
			else
			{
				// TIE
				resultOut = RESULT_TIE;
				return true;
			}
		}
		else if(!redMoveFound)
		{
			if(blackMoveFound)
			{
				// BLACK WIN
				resultOut = RESULT_BLACK_WIN;
				return true;
			}
			else
			{
				// TIE
				resultOut = RESULT_TIE;
				return true;
			}
		}
	}
	resultOut = blackCount - redCount;
	return false;
}

size_t AIUtility::selectResult(std::vector<result_t>* results)
{
	// Get max element
	result_t max_element = *std::max_element(results->begin(), results->end());

	// Get all max element indices
	std::vector<size_t> indices;
	for(size_t i = 0; i < results->size(); i++)
	{
		if(results->at(i) == max_element) indices.push_back(i);
	}

	// Pick random move and return index
	return indices.at((rand() % indices.size()));
}

result_t AIUtility::evalBlackMove(GameBoard board, Move& move, depth_t depth)
{
	// Execute Move
	board.move(move.oldPos, move.newPos);
	if(MOVE_ISJUMP(move)) board.makeEmpty(move.jumpPos);

	// Check for king
	if(move.newPos > 27)
	{
		board.kingPiece(move.newPos);
	}

	TableEntry* entry = blackTable.getEntry(board.getOccupiedBitBoard());
	result_t result;
	if(entry->isMatchResult(board, depth, result)) return result;

	// Check depth and evaluate
	if(evalBoardResult(board, result) || depth == NODE_DEPTH_MINIMAX)
	{
		// Store table result
		entry->replaceResult(board, depth, result);

		return result;
	}

	std::vector<Move>* moves;
	std::vector<result_t> results;
	size_t a;
	if(move.moveType == MOVE_JUMP_MULTI)
	{
		// Create moves
		moves = getAllBlackJumps(board, move.newPos);

		// Evaluate Moves (recursive)
		for(uint8_t i = 0; i < moves->size(); i++)
		{
			results.push_back(evalBlackMove(board, moves->at(i), depth + 1));
		}
		// Pick max result
		auto iterator = std::max_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	else
	{
		// Create moves
		moves = getAllRedMoves(board);

		// Evaluate Moves (recursive)
		for(uint8_t i = 0; i < moves->size(); i++)
		{
			results.push_back(evalRedMove(board, moves->at(i), depth + 1));
		}
		// Pick min result
		auto iterator = std::min_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	delete moves;
	result = results[a];

	// Store table result
	entry->replaceResult(board, depth, result);

	return result;
}

result_t AIUtility::evalRedMove(GameBoard board, Move& move, depth_t depth)
{
	// Execute Move
	board.move(move.oldPos, move.newPos);
	if(MOVE_ISJUMP(move)) board.makeEmpty(move.jumpPos);

	// Check for king
	if(move.newPos < 4)
	{
		board.kingPiece(move.newPos);
	}

	TableEntry* entry = redTable.getEntry(board.getOccupiedBitBoard());

	result_t result;
	if(entry->isMatchResult(board, depth, result)) return result;

	// Check depth and evaluate
	if(evalBoardResult(board, result) || depth == NODE_DEPTH_MINIMAX)
	{
		// Store table result
		entry->replaceResult(board, depth, result);
		return result;
	}

	std::vector<Move>* moves;
	std::vector<result_t> results;
	size_t a;
	if(move.moveType == MOVE_JUMP_MULTI)
	{
		// Create moves
		moves = getAllRedJumps(board, move.newPos);

		// Evaluate Moves (recursive)
		for(uint8_t i = 0; i < moves->size(); i++)
		{
			results.push_back(evalRedMove(board, moves->at(i), depth + 1));
		}
		// Pick min result
		auto iterator = std::min_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	else
	{
		// Create moves
		moves = getAllBlackMoves(board);

		// Evaluate Moves (recursive)
		for(uint8_t i = 0; i < moves->size(); i++)
		{
			results.push_back(evalBlackMove(board, moves->at(i), depth + 1));
		}
		// Pick max result
		auto iterator = std::max_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	delete moves;
	result = results[a];

	// Store table result
	entry->replaceResult(board, depth, result);

	return result;
}
