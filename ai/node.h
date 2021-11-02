#ifndef NODE_H
#define NODE_H

#include "defines.h"

struct Node{};

struct NodeLink : Node
{
	std::vector<Node*> children;
};

struct NodeEnd : Node
{
	result_t result;
};

#endif // NODE_H
