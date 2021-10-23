#ifndef GAMEVIEW_H
#define GAMEVIEW_H

#include <QGraphicsView>
#include <QObject>
#include <QLabel>
#include "gameengine.h"

#define BOARD_VIEW_SIZE 512
#define BOARD_VIEW_STEP (BOARD_VIEW_SIZE / 8)
#define BOARD_VIEW_OFFSET (BOARD_VIEW_STEP / 8)
#define BOARD_VIEW_SCALE (BOARD_VIEW_STEP * 3/4)

class GameView : public QGraphicsView
{
	Q_OBJECT
public:
	explicit GameView(QWidget *parent = nullptr);
	~GameView();

protected:
	QSize sizeHint() const override;
	void resizeEvent(QResizeEvent*) override;
	void drawBackground(QPainter *painter, const QRectF &rect) override;
	void mousePressEvent(QMouseEvent *event) override;

private:
	void drawCheckers();
	void updateBoardSquare(boardpos_t position, SquareState state);

	QGraphicsEllipseItem* checkers[32] = {};

	QGraphicsScene* scene;

	GameEngine gameEngine;

	mutable QSize lastSize;
	QPen redPen;
	QPen blackPen;
	QBrush redBrush;
	QBrush blackBrush;

	QLabel* kingLabel;
};

#endif // GAMEVIEW_H
