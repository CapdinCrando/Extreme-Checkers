#ifndef AIUTILITY_H
#define AIUTILITY_H

#include "../defines.h"
#include "../gameboard.h"

class AIUtility
{
public:
	static std::vector<Move>* getAllBlackMoves(GameBoard &board);
	static std::vector<Move>* getAllBlackJumps(GameBoard &board, boardpos_t pos);
	static std::vector<Move>* getAllRedMoves(GameBoard &board);
	static std::vector<Move>* getAllRedJumps(GameBoard &board, boardpos_t pos);
	static bool evalBoardResult(GameBoard &board, result_t& resultOut);
	static result_t evalBlackMove(GameBoard board, Move& move, depth_t depth);
	static result_t evalRedMove(GameBoard board, Move& move, depth_t depth);
};

#endif // AIUTILITY_H
