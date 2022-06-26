#ifndef TABLE_H
#define TABLE_H

#define TABLE_SIZE 40000000UL

#include "defines.h"
#include "tableentry.h"

class Table
{
public:
	Table();
	TableEntry* getEntry(bitboard_t key);

private:
	TableEntry table[TABLE_SIZE];
};

#endif // TABLE_H
