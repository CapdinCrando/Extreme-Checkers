#ifndef FAKECHECKERITEM_H
#define FAKECHECKERITEM_H

#include "checkeritem.h"

const QBrush fakeRedBrush = QBrush((QColor(255,0,0,50)));
class FakeCheckerItem : public CheckerItem
{
	Q_OBJECT
public:
	FakeCheckerItem(Move move, SquareState checkerType);

signals:
	void fakeCheckerSelected(Move move);

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;

private:
	Move move;
};

#endif // FAKECHECKERITEM_H
