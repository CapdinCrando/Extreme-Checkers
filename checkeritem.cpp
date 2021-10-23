#include "checkeritem.h"

#define BOARD_VIEW_SIZE 512
#define BOARD_VIEW_STEP (BOARD_VIEW_SIZE / 8)
#define BOARD_VIEW_OFFSET (BOARD_VIEW_STEP / 8)
#define BOARD_VIEW_SCALE (BOARD_VIEW_STEP * 3/4)
#define BOARD_VIEW_X(position) BOARD_VIEW_STEP*((position*2 + 1)%8 - ((position/4)%2)) + BOARD_VIEW_OFFSET
#define BOARD_VIEW_Y(position) BOARD_VIEW_STEP*( position/4) + BOARD_VIEW_OFFSET

CheckerItem::CheckerItem(boardpos_t position, SquareState checkerType)
	: QGraphicsEllipseItem(BOARD_VIEW_X(position), BOARD_VIEW_Y(position), BOARD_VIEW_SCALE, BOARD_VIEW_SCALE)
{
	// Set position
	this->position = position;

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

	// Set normal or king
	if(SQUARE_ISKING(checkerType))
	{
		QGraphicsSimpleTextItem* kingLabel = new QGraphicsSimpleTextItem("K");
		kingLabel->setBrush(whiteBrush);
		kingLabel->setParentItem(this);
		kingLabel->setPos(this->boundingRect().center());
	}
}

void CheckerItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	Q_UNUSED(event);
	checkerSelected(this->position);
}
