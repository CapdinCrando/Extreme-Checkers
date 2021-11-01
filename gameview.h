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

signals:
	void gameOver(GameState gameState);

public slots:
	void onCheckerSelected(boardpos_t pos, SquareState state);
	void drawPossibleMoves(std::vector<Move> moves, SquareState checkerType);
	void startRedMove(Move move);
	void blackMoveFinished();
	void displayMove(Move move, bool kingPiece);
	void resetBoard();
	void saveSettings(GameSettings settings);

protected:
	QSize sizeHint() const override;
	void resizeEvent(QResizeEvent*) override;
	void drawBackground(QPainter *painter, const QRectF &rect) override;
	void mousePressEvent(QMouseEvent *event) override;

private:
	void clearFakeCheckers();

	CheckerItem* checkers[32] = {};
	std::vector<FakeCheckerItem*> fakeItems;

	QGraphicsScene* scene;

	GameEngine gameEngine;

	mutable QSize lastSize;

	bool acceptingClicks = false;
};

#endif // GAMEVIEW_H
