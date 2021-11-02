#ifndef NODE_H
#define NODE_H

#include "defines.h"

struct Node
{
	result_t result;
	std::vector<Node*> children;
};

#endif // NODE_H
