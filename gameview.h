#ifndef GAMEVIEW_H
#define GAMEVIEW_H

#include <QGraphicsView>
#include <QObject>
#include <QLabel>
#include "gameengine.h"

class GameView : public QGraphicsView
{
	Q_OBJECT
public:
	explicit GameView(QWidget *parent = nullptr);
	~GameView();

public slots:
	//void boardUpdated(BoardState b);

protected:
	QSize sizeHint() const override;
	void resizeEvent(QResizeEvent*) override;
	void drawBackground(QPainter *painter, const QRectF &rect) override;

private:
	void drawCheckers();
	void drawChecker(int x, int y, int w, int h, SquareState state);
	void clearSquare(boardpos_t position);

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
