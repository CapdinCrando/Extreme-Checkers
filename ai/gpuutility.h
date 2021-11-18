#ifndef GPUUTILITY_H
#define GPUUTILITY_H

#include "../defines.h"
#include "../gameboard.h"

typedef int result_gpu_t;

class GPUUtility
{
public:
	static void initializeGPU();
	static Move getMove(GameBoard board);
	static void clear();

private:
	static Move *moves_dev;
	static boardstate_t *board_dev;
	static result_gpu_t *results_dev;
};

#endif // GPUUTILITY_H
