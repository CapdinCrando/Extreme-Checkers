#include "fakecheckeritem.h"

#include <iostream>

FakeCheckerItem::FakeCheckerItem(Move move, SquareState checkerType) : CheckerItem(move.newPos, checkerType)
{
	this->setBrush(fakeRedBrush);
	this->move = move;
}

void FakeCheckerItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	Q_UNUSED(event);
	emit fakeCheckerSelected(this->move);
}
