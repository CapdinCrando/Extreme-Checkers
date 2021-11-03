#include "aimanager.h"
#include "airandom.h"
#include "aiminimax.h"
#include "aiparallel.h"
#include "aigpu.h"

#ifdef QT_DEBUG
#include <iostream>
#include <chrono>
#endif

const std::vector<AI*> AIManager::aiList = { new AIRandom(),
											 new AIMinimax(),
											 new AIParallel(),
											 new AIGPU() };

AIManager::~AIManager()
{
	for(uint8_t i = 0; i < aiList.size(); i++) delete aiList[i];
}

QStringList AIManager::getDescriptionList()
{
	QStringList descriptionList;
	for(uint8_t i = 0; i < aiList.size(); i++) descriptionList.append(aiList[i]->getDescription());
	return descriptionList;
}


void AIManager::selectAI(uint8_t index)
{
	if(aiList.size() > index)
	{
		currentAI = aiList[index];
	}
}

Move AIManager::getMove(GameBoard board)
{
#ifdef QT_DEBUG
	auto start = std::chrono::high_resolution_clock::now();
	Move move = currentAI->getMove(board);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Black move calculated in " << duration.count() << " us" << std::endl;
	return move;
#else
	return currentAI->getMove(board);
#endif
}

