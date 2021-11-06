#ifndef GPUUTILITY_H
#define GPUUTILITY_H

#include "../defines.h"
#include "../gameboard.h"

class GPUUtility
{
public:
	static void initializeGPU();
	static Move getMove(BoardState* board);
	static void clear();

private:
	static Move *move_host, *move_dev;
	static boardstate_t *board_dev;
};

#endif // GPUUTILITY_H
