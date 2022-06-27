#include "aitask.h"
#include "aiutility.h"

void AITask::run()
{
	resultOut = AIUtility::evalBlackMove(board, move, 0, RESULT_MIN, RESULT_MAX);
}
