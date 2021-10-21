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

void GameView::updateBoardSquare(boardpos_t position, SquareState state)
{
	if(SQUARE_ISEMPTY(state))
	{
		QGraphicsEllipseItem* checker = checkers[position];
		if(checker != nullptr)
		{
			scene->removeItem(checker);
			checkers[position] = nullptr;
		}
	}
	else
	{
		if(checkers[position] == nullptr)
		{
			int y_scale = position/4;
			int x = BOARD_VIEW_STEP*((position*2 + 1)%8 - (y_scale%2)) + BOARD_VIEW_OFFSET;
			int y = BOARD_VIEW_STEP*(y_scale) + BOARD_VIEW_OFFSET;

			QGraphicsEllipseItem* checker;
			if(SQUARE_ISBLACK(state)) checker = scene->addEllipse(x, y, BOARD_VIEW_SCALE, BOARD_VIEW_SCALE, redPen, blackBrush);
			else checker = scene->addEllipse(x, y, BOARD_VIEW_SCALE, BOARD_VIEW_SCALE, blackPen, redBrush);
			if(SQUARE_ISKING(state))
			{
				QGraphicsProxyWidget *proxyWidget = new QGraphicsProxyWidget(checker);
				proxyWidget->setWidget(kingLabel);
				proxyWidget->setPos(checker->boundingRect().center()-kingLabel->rect().center());
			}
			checkers[position] = checker;
		}
	}
}

void GameView::drawCheckers()
{
	for(boardpos_t k = 0; k < SQUARE_COUNT; k++)
	{
		updateBoardSquare(k, gameEngine.getSquareState(k));
	}
}

