#include "ai.h"

std::vector<Move>* AI::getAllBlackMoves(GameBoard &board)
{
	Move m;
	std::vector<Move> *moves = new std::vector<Move>;
	std::vector<Move> *jumps = new std::vector<Move>;
	for(uint8_t i = 0; i < SQUARE_COUNT; i++)
	{
		SquareState state = board.getSquareState(i);
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
					SquareState moveState = board.getSquareState(move);
					if(SQUARE_ISEMPTY(moveState))
					{
						// Add move to potential moves
						m.oldPos = i;
						m.newPos = move;
						m.moveType = MOVE_MOVE;
						moves->push_back(m);
					}
					else if(!(SQUARE_ISBLACK(moveState)))
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
								for(uint8_t k = cornerMin; k < 4; k++)
								{
									boardpos_t moveMulti = cornerList[j][k];
									// Check if position is invalid
									if(moveMulti != BOARD_POS_INVALID)
									{
										SquareState moveStateMulti = board.getSquareState(moveMulti);
										if(SQUARE_ISNOTEMPTY(moveStateMulti))
										{
											if(!(SQUARE_ISBLACK(moveStateMulti)))
											{
												boardpos_t jumpMulti = cornerList[j][k];
												if(jumpMulti != BOARD_POS_INVALID)
												{
													SquareState jumpStateMulti = board.getSquareState(moveMulti);
													if(SQUARE_ISNOTEMPTY(jumpStateMulti))
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
