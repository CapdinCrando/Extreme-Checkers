#ifndef TABLEENTRY_H
#define TABLEENTRY_H

#include "gameboard.h"
#include <mutex>
#include "defines.h"


class TableEntry
{
public:
	TableEntry();
	bool isMatchResult(GameBoard& board, depth_t depth, result_t& result);
	void replaceResult(GameBoard& board, depth_t depth, result_t& result);

private:
	std::mutex mutex;
	GameBoard board;
	depth_t depth = -1;
	result_t result = RESULT_TIE;
};

#endif // TABLEENTRY_H
