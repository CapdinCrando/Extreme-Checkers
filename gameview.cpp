#include "gameview.h"
#include <iostream>
#include <QGraphicsPixmapItem>
#include <QGraphicsProxyWidget>

GameView::GameView(QWidget *parent) : QGraphicsView(parent)
{
	this->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	this->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

	scene = new QGraphicsScene();
	this->setScene(scene);
	this->scene->setSceneRect(0, 0, BOARD_VIEW_SIZE, BOARD_VIEW_SIZE);

	redPen = QPen(Qt::red);
	blackPen = QPen(Qt::black);
	redBrush = QBrush(Qt::red);
	blackBrush = QBrush(Qt::black);

	kingLabel = new QLabel();
	kingLabel->setText("K");

	gameEngine.resetGame();

	drawCheckers();

	this->show();
}

GameView::~GameView()
{
	delete scene;
}

QSize GameView::sizeHint() const
{
	QSize s = size();
	lastSize = s;
	s.setWidth(s.height());
	s.setHeight(QGraphicsView::sizeHint().height());
	return s;
}

void GameView::resizeEvent(QResizeEvent* event)
{
	QGraphicsView::resizeEvent(event);
	int h = height();
	int w = width();
	if(w > h)
	{
		if (lastSize.height()!=h) {
			this->resize(h, h);
		}
	}
	else if(w < h)
	{
		if (lastSize.width()!=w) {
			this->resize(w, w);
		}
	}
	QTransform Matrix(1, 0, 0, 0, 1, 0, 0, 0, 1);
	Matrix.scale(width() / sceneRect().width(), height() / sceneRect().height());
	setTransform(Matrix);
}

void GameView::drawBackground(QPainter *painter, const QRectF &rect)
{
	QPixmap pixmap(":/img/board.png");
	pixmap = pixmap.scaled(this->size(), Qt::KeepAspectRatioByExpanding);
	painter->drawPixmap(rect, pixmap, QRectF(pixmap.rect()));
}

void GameView::drawCheckers()
{
	int step = BOARD_VIEW_SIZE/8;
	int offset = step/8;
	int scale = step*3/4;
	for(boardpos_t k = 0; k < SQUARE_COUNT; k++)
	{
		SquareState state = gameEngine.getSquareState(k);
		if(SQUARE_ISNOTEMPTY(state))
		{
			int y = k/4;
			drawChecker(step*((k*2 + 1)%8 - (y%2)) + offset, step*(y) + offset, scale, scale, state);
		}
	}
}

void GameView::drawChecker(int x, int y, int w, int h, SquareState state)
{
	QPainter p(this);
	QGraphicsEllipseItem* checker;
	if(SQUARE_ISBLACK(state)) checker = scene->addEllipse(x, y, w, h, redPen, blackBrush);
	else checker = scene->addEllipse(x, y, w, h, blackPen, redBrush);
	if(SQUARE_ISKING(state))
	{
		QGraphicsProxyWidget *proxyWidget = new QGraphicsProxyWidget(checker);
		proxyWidget->setWidget(kingLabel);
		proxyWidget->setPos(checker->boundingRect().center()-kingLabel->rect().center());
	}
}
