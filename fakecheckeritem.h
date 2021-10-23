#ifndef FAKECHECKERITEM_H
#define FAKECHECKERITEM_H

#include "checkeritem.h"

const QBrush fakeRedBrush = QBrush((QColor(255,0,0,50)));
class FakeCheckerItem : public CheckerItem
{
	Q_OBJECT
public:
	FakeCheckerItem(boardpos_t position, SquareState checkerType);
};

#endif // FAKECHECKERITEM_H
