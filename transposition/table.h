#ifndef TABLE_H
#define TABLE_H

#define TABLE_SIZE 20000000UL

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
