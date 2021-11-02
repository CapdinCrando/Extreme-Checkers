#include "checkeritem.h"

#include <iostream>

#define BOARD_VIEW_SIZE 512
#define BOARD_VIEW_STEP (BOARD_VIEW_SIZE / 8)
#define BOARD_VIEW_OFFSET (BOARD_VIEW_STEP / 8)
#define BOARD_VIEW_SCALE (BOARD_VIEW_STEP * 3/4)
#define BOARD_VIEW_X(position) BOARD_VIEW_STEP*((position*2 + 1)%8 - ((position/4)%2)) + BOARD_VIEW_OFFSET
#define BOARD_VIEW_Y(position) BOARD_VIEW_STEP*( position/4) + BOARD_VIEW_OFFSET

#define LABEL_X 10
#define LABEL_Y 20

CheckerItem::CheckerItem(boardpos_t position, SquareState checkerType)
	: QGraphicsEllipseItem(0, 0, BOARD_VIEW_SCALE, BOARD_VIEW_SCALE)
{
	// Set position
	this->position = position;
	this->setPos(BOARD_VIEW_X(position), BOARD_VIEW_Y(position));

	// Set checker type
	this->checkerType = checkerType;

	// Set red or black
	if(SQUARE_ISBLACK(checkerType))
	{
		this->setPen(blackPen);
		this->setBrush(blackBrush);
	}
	else
	{
		this->setPen(redPen);
		this->setBrush(redBrush);
	}
}

void CheckerItem::move(boardpos_t newPosition)
{
	this->position = newPosition;
	this->setPos(BOARD_VIEW_X(newPosition), BOARD_VIEW_Y(newPosition));
}

void CheckerItem::king()
{
	// Set checker type
	if(SQUARE_ISBLACK(checkerType)) checkerType = SQUARE_BLACK_KING;
	else checkerType = SQUARE_RED_KING;

	// Set label
	QGraphicsSimpleTextItem* kingLabel = new QGraphicsSimpleTextItem("K");
	kingLabel->setBrush(whiteBrush);
	kingLabel->setParentItem(this);
	QPointF checkerCenter = this->boundingRect().center();
	QRectF labelBox = kingLabel->boundingRect();
	kingLabel->setPos(checkerCenter.x() - labelBox.width(), checkerCenter.y() - labelBox.height());
	kingLabel->setScale(2);
}

void CheckerItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	Q_UNUSED(event);
	emit checkerSelected(this->position, this->checkerType);
}
