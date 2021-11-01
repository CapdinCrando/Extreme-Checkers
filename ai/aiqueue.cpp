#include "aiqueue.h"

AIQueue::AIQueue()
{

}

void AIQueue::push(Node n)
{
	mutex.lock();
	std::queue<Node>::push(n);
	mutex.unlock();
}

Node AIQueue::pop()
{
	mutex.lock();
	Node n = std::queue<Node>::front();
	std::queue<Node>::pop();
	mutex.unlock();
	return n;
}
