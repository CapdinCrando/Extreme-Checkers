#include "gameview.h"
#include <iostream>
#include <QGraphicsPixmapItem>
#include <QGraphicsProxyWidget>

GameView::GameView(QWidget *parent) : QGraphicsView(parent)
{
	this->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	this->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	this->setAutoFillBackground(true);
	this->setCacheMode(QGraphicsView::CacheBackground);
	this->setAlignment(Qt::AlignLeft | Qt::AlignTop);

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

	resetBoard();

	this->show();
	acceptingClicks = true;
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

void GameView::mousePressEvent(QMouseEvent *event)
{
	if(acceptingClicks) QGraphicsView::mousePressEvent(event);
	else event->ignore();
}

void GameView::clearFakeCheckers()
{
	for(uint8_t i = 0; i < fakeItems.size(); i++)
	{
		scene->removeItem(fakeItems[i]);
	}
	fakeItems.clear();
}

void GameView::drawFakeCheckers(boardpos_t pos, SquareState checkerType)
{
	// Clear old items
	this->clearFakeCheckers();

	// Get moves
	std::vector<Move> moves = gameEngine.getPossibleMoves(pos);

	// Add new items
	for(uint8_t i = 0; i < moves.size(); i++)
	{
		FakeCheckerItem* fakeChecker = new FakeCheckerItem(moves[i], checkerType);
		scene->addItem(fakeChecker);
		connect(fakeChecker, &FakeCheckerItem::fakeCheckerSelected, this, &GameView::displayMove);
		fakeItems.push_back(fakeChecker);
	}
}

void GameView::displayMove(Move move)
{
	this->acceptingClicks = false;
	this->clearFakeCheckers();
	gameEngine.move(move);
	CheckerItem* checker = checkers[move.oldPos];
	checker->move(move.newPos);
	checkers[move.newPos] = checker;
	this->acceptingClicks = true;
}

void GameView::updateBoardSquare(boardpos_t position, SquareState state)
{
	CheckerItem* checker = checkers[position];
	if(SQUARE_ISEMPTY(state))
	{
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
			checker = new CheckerItem(position, state);
			if(!(SQUARE_ISBLACK(state)))
			{
				connect(checker, &CheckerItem::checkerSelected, this, &GameView::drawFakeCheckers);
			}
			scene->addItem(checker);
			checkers[position] = checker;
		}
	}
}

void GameView::resetBoard()
{
	for(boardpos_t k = 0; k < SQUARE_COUNT; k++)
	{
		updateBoardSquare(k, gameEngine.getSquareState(k));
	}
}

