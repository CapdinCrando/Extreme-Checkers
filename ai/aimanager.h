#ifndef AIMANAGER_H
#define AIMANAGER_H

#include <vector>
#include <QStringList>
#include "ai.h"

class AIManager
{

public:
	~AIManager();
	static QStringList getDescriptionList();
	void selectAI(uint8_t index);
	Move getMove(GameBoard board);

private:
	static const std::vector<AI*> aiList;
	AI* currentAI = aiList[0];
};

#endif // AIMANAGER_H
