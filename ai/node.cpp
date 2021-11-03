#include "node.h"

Node::~Node()
{
	for(uint8_t i = 0; i < children.size(); i++)
	{
		delete children.at(i);
	}
}
