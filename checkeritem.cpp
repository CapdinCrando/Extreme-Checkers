#include "checkeritem.h"

#define BOARD_VIEW_SIZE 512
#define BOARD_VIEW_STEP (BOARD_VIEW_SIZE / 8)
#define BOARD_VIEW_OFFSET (BOARD_VIEW_STEP / 8)
#define BOARD_VIEW_SCALE (BOARD_VIEW_STEP * 3/4)

const QPen CheckerItem::redPen = QPen(Qt::red);
const QPen CheckerItem::blackPen = QPen(Qt::black);
const QBrush CheckerItem::redBrush = QBrush(Qt::red);
const QBrush CheckerItem::blackBrush = QBrush(Qt::black);

CheckerItem::CheckerItem(boardpos_t position) : QGraphicsEllipseItem()
{
	int y_scale = position/4;
	int x = BOARD_VIEW_STEP*((position*2 + 1)%8 - (y_scale%2)) + BOARD_VIEW_OFFSET;
	int y = BOARD_VIEW_STEP*(y_scale) + BOARD_VIEW_OFFSET;

	//QGraphicsEllipseItem::QGraphicsEllipseItem(x, y, BOARD_VIEW_SCALE, BOARD_VIEW_SCALE);
	//else checker = scene->addEllipse(x, y, BOARD_VIEW_SCALE, BOARD_VIEW_SCALE, blackPen, redBrush);

}
