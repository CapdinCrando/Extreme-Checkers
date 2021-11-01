#ifndef AIQUEUE_H
#define AIQUEUE_H

#include <cinttypes>
#include <queue>
#include <mutex>

struct Node
{
	uint8_t data;
};

class AIQueue : protected std::queue<Node>
{
public:
	AIQueue();
	void push(Node n);
	Node pop();

private:
	std::mutex mutex;
};

#endif // AIQUEUE_H
