#include "aimanager.h"

const std::vector<AI*> AIManager::aiList = { new AIRandom() };

AIManager::~AIManager()
{
	for(uint8_t i = 0; i < aiList.size(); i++) delete aiList[i];
	aiList.empty();
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

AIMove AIManager::getMove()
{
	return currentAI->getMove();
}

