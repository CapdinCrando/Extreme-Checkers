#ifndef GAMEWINDOW_H
#define GAMEWINDOW_H

#include <QWidget>
#include <QPainter>
#include "gameengine.h"

class GameWindow : public QWidget
{
	Q_OBJECT

public:
	explicit GameWindow(QWidget *parent = nullptr);
	~GameWindow();

public slots:
	void boardUpdated(BoardState b);

protected:
	void paintEvent(QPaintEvent* event) override;
	QSize sizeHint() const override;
	void resizeEvent(QResizeEvent*) override;

private:
	void drawCheckers(int n);
	void drawChecker(int x, int y, int w, int h, SquareState state);
	void clearSquare(boardpos_t position);

	GameEngine gameEngine;

	mutable QSize lastSize;
	QPen pen;
	QBrush brush;
};

#endif // GAMEWINDOW_H
