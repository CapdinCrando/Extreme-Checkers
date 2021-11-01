#include "aiminimax.h"

#define NODE_DEPTH 5

std::vector<Move>* AIMinimax::getAllRedMoves(GameBoard &board)
{
	Move m;
	std::vector<Move> *moves = new std::vector<Move>;
	std::vector<Move> *jumps = new std::vector<Move>;
	for(uint8_t i = 0; i < SQUARE_COUNT; i++)
	{
		SquareState state = board.getSquareState(i);
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
					SquareState moveState = board.getSquareState(move);
					if(SQUARE_ISEMPTY(moveState))
					{
						// Add move to potential moves
						m.oldPos = i;
						m.newPos = move;
						m.moveType = MOVE_MOVE;
						moves->push_back(m);
					}
					else if(SQUARE_ISBLACK(moveState))
					{
						// Get jump
						boardpos_t jump = cornerList[move][j];
						// Check if position is invalid
						if(jump != BOARD_POS_INVALID)
						{
							// Check if space is empty
							if(SQUARE_ISEMPTY(board.getSquareState(jump)))
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
											SquareState moveStateMulti = board.getSquareState(moveMulti);
											if(SQUARE_ISNOTEMPTY(moveStateMulti))
											{
												if(SQUARE_ISBLACK(moveStateMulti))
												{
													boardpos_t jumpMulti = cornerList[moveMulti][k];
													if(jumpMulti != BOARD_POS_INVALID)
													{
														SquareState jumpStateMulti = board.getSquareState(jumpMulti);
														if(SQUARE_ISEMPTY(jumpStateMulti))
														{
															m.moveType = MOVE_JUMP_MULTI;
															break;
														}
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

std::vector<Move>* AIMinimax::getAllRedJumps(GameBoard &board, boardpos_t pos)
{
	Move m;
	std::vector<Move> *jumps = new std::vector<Move>;
	SquareState state = board.getSquareState(pos);
	if(SQUARE_ISNOTEMPTY(state))
	{
		for(uint8_t j = 0; j < 4; j++)
		{
			// Get move
			boardpos_t move = cornerList[pos][j];
			// Check if position is invalid
			if(move != BOARD_POS_INVALID)
			{
				// Check if space is empty
				SquareState moveState = board.getSquareState(move);
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
							if(SQUARE_ISEMPTY(board.getSquareState(jump)))
							{
								// Add move to potential moves
								m.oldPos = pos;
								m.newPos = jump;
								m.jumpPos = move;

								// Check for multi
								m.moveType = MOVE_JUMP;
								for(uint8_t k = 0; k < 4; k++)
								{
									boardpos_t moveMulti = cornerList[move][k];
									// Check if position is invalid
									if(moveMulti != BOARD_POS_INVALID)
									{
										if(moveMulti != move)
										{
											SquareState moveStateMulti = board.getSquareState(moveMulti);
											if(SQUARE_ISNOTEMPTY(moveStateMulti))
											{
												if(SQUARE_ISBLACK(moveStateMulti))
												{
													boardpos_t jumpMulti = cornerList[moveMulti][k];
													if(jumpMulti != BOARD_POS_INVALID)
													{
														SquareState jumpStateMulti = board.getSquareState(jumpMulti);
														if(SQUARE_ISEMPTY(jumpStateMulti))
														{
															m.moveType = MOVE_JUMP_MULTI;
															break;
														}
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
	return jumps;
}

bool AIMinimax::evalBoardResult(GameBoard &board, result_t& resultOut)
{
	result_t blackCount = 0;
	result_t redCount = 0;
	bool redMoveFound = false;
	bool blackMoveFound = false;
	for(boardpos_t pos = 0; pos < SQUARE_COUNT; pos++)
	{
		SquareState state = board.getSquareState(pos);
		if(SQUARE_ISNOTEMPTY(state))
		{
			if(SQUARE_ISBLACK(state))
			{
				blackCount += 1;
				uint8_t cornerMin = 2;
				if(SQUARE_ISKING(state)) cornerMin = 0;
				for(uint8_t i = cornerMin; i < 4; i++)
				{
					// Get move
					boardpos_t move = cornerList[pos][i];

					// Check if position is invalid
					if(move != BOARD_POS_INVALID)
					{
						// Check if space is empty
						SquareState moveState = board.getSquareState(move);
						if(SQUARE_ISEMPTY(moveState))
						{
							blackMoveFound = true;
						}
						else if(!(SQUARE_ISBLACK(moveState)))
						{
							// Get jump
							boardpos_t jump = cornerList[move][i];
							// Check if position is invalid
							if(jump != BOARD_POS_INVALID)
							{
								// Check if space is empty
								if(SQUARE_ISEMPTY(board.getSquareState(jump)))
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
				if(SQUARE_ISKING(state)) cornerMax = 4;
				for(uint8_t i = 0; i < cornerMax; i++)
				{
					// Get move
					boardpos_t move = cornerList[pos][i];

					// Check if position is invalid
					if(move != BOARD_POS_INVALID)
					{
						// Check if space is empty
						SquareState moveState = board.getSquareState(move);
						if(SQUARE_ISEMPTY(moveState))
						{
							redMoveFound = true;
						}
						else if(SQUARE_ISBLACK(moveState))
						{
							// Get jump
							boardpos_t jump = cornerList[move][i];
							// Check if position is invalid
							if(jump != BOARD_POS_INVALID)
							{
								// Check if space is empty
								if(SQUARE_ISEMPTY(board.getSquareState(jump)))
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

result_t AIMinimax::evalBlackMove(GameBoard board, Move& move, depth_t depth)
{
	// Execute Move
	board.move(move.oldPos, move.newPos);
	if(MOVE_ISJUMP(move)) board.setSquareState(move.jumpPos, SQUARE_EMPTY);

	// Check depth and evaluate
	result_t result;
	if(evalBoardResult(board, result)) return result;
	else if(depth == NODE_DEPTH) return result;

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
			results.push_back(evalBlackMove(board, moves->at(i), depth++));
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
			results.push_back(evalRedMove(board, moves->at(i), depth++));
		}
		// Pick min result
		auto iterator = std::min_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	return results[a];
}

result_t AIMinimax::evalRedMove(GameBoard board, Move& move, depth_t depth)
{
	// Execute Move
	board.move(move.oldPos, move.newPos);
	if(MOVE_ISJUMP(move)) board.setSquareState(move.jumpPos, SQUARE_EMPTY);

	// Check depth and evaluate
	result_t result;
	if(evalBoardResult(board, result)) return result;
	else if(depth == NODE_DEPTH) return result;

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
			results.push_back(evalRedMove(board, moves->at(i), depth++));
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
			results.push_back(evalBlackMove(board, moves->at(i), depth++));
		}
		// Pick max result
		auto iterator = std::max_element(std::begin(results), std::end(results));
		a = std::distance(results.begin(), iterator);
	}
	return results[a];
}

Move AIMinimax::getMove(GameBoard& board)
{
	std::vector<Move>* moves = getAllBlackMoves(board);
	if(moves->empty())
	{
		Move m;
		m.moveType = MOVE_INVALID;
		return m;
	}

	std::vector<result_t> results;

	for(uint8_t i = 0; i < moves->size(); i++)
	{
		results.push_back(evalBlackMove(board, moves->at(i), 0));
	}

	// Pick result
	auto iterator = std::max_element(std::begin(results), std::end(results));
	size_t a = std::distance(results.begin(), iterator);
	return moves->at(a);
}
