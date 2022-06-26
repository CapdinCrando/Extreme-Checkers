#include "table.h"

Table::Table()
{

}

TableEntry* Table::getEntry(bitboard_t key)
{
	uint32_t index = key % TABLE_SIZE;
	return &table[index];
}
