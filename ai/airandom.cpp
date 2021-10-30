#include "airandom.h"

AIRandom::AIRandom(QObject *parent) : AI(parent)
{

}

AIMove AIRandom::getMove()
{
	return AIMove();
}
