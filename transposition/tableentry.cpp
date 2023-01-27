#include "tableentry.h"

TableEntry::TableEntry()
{

}


bool TableEntry::isMatchResult(GameBoard& board, depth_t depth, result_t& result)
{
	bool returnValue = false;
	mutex.lock();
	if(this->board == board && this->depth <= depth)
	{
		result = this->result;
		returnValue = true;
	}
	mutex.unlock();
	return returnValue;
}

void TableEntry::replaceResult(GameBoard& board, depth_t depth, result_t& result)
{
	mutex.lock();
	this->board = board;
	this->depth = depth;
	this->result = result;
	mutex.unlock();
}
