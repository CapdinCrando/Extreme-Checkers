#ifndef CHECKERITEM_H
#define CHECKERITEM_H

#include <QGraphicsEllipseItem>
#include <QObject>
#include <QPen>
#include <QBrush>
#include <QLabel>

#include <gameengine.h>

class CheckerItem : public QGraphicsEllipseItem
{
	Q_OBJECT
public:
	CheckerItem(boardpos_t position);

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;

private:
	boardpos_t position;

	const static QPen redPen;
	const static QPen blackPen;
	const static QBrush redBrush;
	const static QBrush blackBrush;
};

#endif // CHECKERITEM_H
