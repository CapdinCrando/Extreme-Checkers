#include "fakecheckeritem.h"

FakeCheckerItem::FakeCheckerItem(boardpos_t position, SquareState checkerType) : CheckerItem(position, checkerType)
{
	this->setBrush(fakeRedBrush);
}
