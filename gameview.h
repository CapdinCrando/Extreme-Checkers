#ifndef GAMEVIEW_H
#define GAMEVIEW_H

#include <QGraphicsView>
#include <QObject>
#include <QLabel>
#include "gameengine.h"
#include "checkeritem.h"
#include "fakecheckeritem.h"

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

public slots:
	void drawFakeCheckers(boardpos_t pos, SquareState state);
	void displayMove(Move move);

protected:
	QSize sizeHint() const override;
	void resizeEvent(QResizeEvent*) override;
	void drawBackground(QPainter *painter, const QRectF &rect) override;
	void mousePressEvent(QMouseEvent *event) override;

private:
	void resetBoard();
	void clearFakeCheckers();
	void updateBoardSquare(boardpos_t position, SquareState checkerType);

	CheckerItem* checkers[32] = {};
	std::vector<FakeCheckerItem*> fakeItems;

	QGraphicsScene* scene;

	GameEngine gameEngine;

	mutable QSize lastSize;
	QPen redPen;
	QPen blackPen;
	QBrush redBrush;
	QBrush blackBrush;

	QLabel* kingLabel;

	bool acceptingClicks = false;
};

#endif // GAMEVIEW_H
