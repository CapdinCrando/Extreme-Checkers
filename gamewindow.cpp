#include "gamewindow.h"

GameWindow::GameWindow(QWidget *parent) : QWidget(parent)
{
	setAutoFillBackground(true);
	pen.setColor(Qt::red);
	brush.setColor(Qt::red);
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
			this->resize(height(), height()); // it is possible that this call should be scheduled to next iteration of event loop
		}
	}
	else if(width() < height())
	{
		if (lastSize.width()!=width()) {
			this->resize(width(), width()); // it is possible that this call should be scheduled to next iteration of event loop
		}
	}


}

void GameWindow::paintEvent(QPaintEvent *event)
{
	QPainter p(this);
	//QRect rectangle(10, 20, 80, 60);
	//p.fillRect(rectangle, QBrush(Qt::red));
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
	for(uint8_t	i = 0; i < 8; i++)
	{
		for(uint8_t	j = 0; j < 8; j++)
		{
			drawChecker(step*i, step*j, step, step);
		}
	}
}

void GameWindow::drawChecker(int x, int y, int w, int h)
{
	QPainter p(this);
	p.setBrush(Qt::red);
	p.drawEllipse(x, y, w, h);

}

