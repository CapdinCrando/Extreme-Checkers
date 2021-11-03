#ifndef NODE_H
#define NODE_H

#include "defines.h"

class Node
{
public:
	result_t result : 7;
	bool isBlack : 1;
	std::vector<Node*> children;

	~Node();
};

#endif // NODE_H
