#include "gamewindow.h"

#include <iostream>
#include <gameengine.h>

GameWindow::GameWindow(QWidget *parent) : QWidget(parent)
{

	setAutoFillBackground(true);
	pen.setColor(Qt::red);
	brush.setColor(Qt::red);
	gameEngine.resetGame();

	update();
}

GameWindow::~GameWindow()
{

}

QSize GameWindow::sizeHint() const
{
	QSize s = size();
	lastSize = s;
	s.setWidth(s.height());
	s.setHeight(QWidget::sizeHint().height());
	return s;
}

void GameWindow::resizeEvent(QResizeEvent* event)
{
	QWidget::resizeEvent(event);

	if(width() > height())
	{
		if (lastSize.height()!=height()) {
			this->resize(height(), height());
		}
	}
	else if(width() < height())
	{
		if (lastSize.width()!=width()) {
			this->resize(width(), width());
		}
	}
}

void GameWindow::paintEvent(QPaintEvent *event)
{
	Q_UNUSED(event);
	QPainter p(this);
	QPixmap pixmap(":/img/board.png");
	pixmap = pixmap.scaled(this->size(), Qt::KeepAspectRatioByExpanding);
	QPalette palette;
	palette.setBrush(QPalette::Window, pixmap);
	this->setPalette(palette);
	drawCheckers(this->height());
}

void GameWindow::drawCheckers(int n)
{
	int step = n/8;
	int offset = step/8;
	int scale = step*3/4;
	static bool printed = false;
	for(boardpos_t k = 0; k < SQUARE_COUNT; k++)
	{
		SquareState state = gameEngine.getSquareState(k);
		if(SQUARE_ISNOTEMPTY(state))
		{
			int y = k/4;
			drawChecker(step*((k*2 + 1)%8 - (y%2)) + offset, step*(y) + offset, scale, scale, state);
			if(!printed) std::cout << (k*2 + 1)%9 << ", " << k/4 << std::endl;
		}
	}
	printed = true;
}

void GameWindow::drawChecker(int x, int y, int w, int h, SquareState state)
{
	QPainter p(this);
	if(SQUARE_ISBLACK(state)) p.setBrush(Qt::black);
	else p.setBrush(Qt::red);
	p.drawEllipse(x, y, w, h);
	if(SQUARE_ISKING(state))
	{
		p.setBrush(Qt::white);
		p.drawText(x, y, w, h, Qt::AlignCenter, "K");
	}
}

