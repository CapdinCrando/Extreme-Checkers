#include "table.h"

Table::Table()
{

}

TableEntry* Table::getEntry(bitboard_t key)
{
	return &table[key % TABLE_SIZE];
}
