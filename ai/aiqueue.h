#ifndef AIQUEUE_H
#define AIQUEUE_H

#include <cinttypes>
#include <queue>
#include <mutex>

#include "defines.h"
#include "gameboard.h"

struct Node
{
	Move m;
	GameBoard board;
};

struct NodeLink : Node
{
	std::vector<Node*> children;
};

struct NodeEnd : Node
{
	result_t result;
};

class AIQueue : protected std::queue<Node*>
{
public:
	AIQueue();
	void push(Node* n);
	Node* pop();

private:
	std::mutex mutex;
};

#endif // AIQUEUE_H
