#ifndef NODE_H
#define NODE_H

#include "defines.h"

struct Node
{
	result_t result : 7;
	bool isBlack : 1;
	std::vector<Node*> children;
};

#endif // NODE_H
