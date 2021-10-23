#ifndef CHECKERITEM_H
#define CHECKERITEM_H

#include <QGraphicsEllipseItem>
#include <QObject>
#include <QPen>
#include <QBrush>
#include <QLabel>

#include <gameengine.h>

const QPen redPen = QPen(Qt::red);
const QBrush redBrush = QBrush(Qt::red);
const QPen blackPen = QPen(Qt::black);
const QBrush blackBrush = QBrush(Qt::black);

class CheckerItem : public QGraphicsEllipseItem
{
	Q_OBJECT
public:
	CheckerItem(boardpos_t position, SquareState checkerType);

signals:
	void checkerSelected(boardpos_t pos);

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;

private:
	boardpos_t position;
	SquareState checkerType;
};

#endif // CHECKERITEM_H
