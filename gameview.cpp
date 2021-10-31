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

	connect(&gameEngine, &GameEngine::blackMoveFinished, this, &GameView::blackMoveFinished);
	connect(&gameEngine, &GameEngine::displayMove, this, &GameView::displayMove);
	connect(&gameEngine, &GameEngine::displayMultiJump, this, &GameView::drawPossibleMoves);
	connect(&gameEngine, &GameEngine::gameOver, this, &GameView::gameOver);

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

void GameView::mousePressEvent(QMouseEvent *event)
{
	if(acceptingClicks) QGraphicsView::mousePressEvent(event);
	else event->ignore();
}

void GameView::saveSettings(GameSettings settings)
{
	acceptingClicks = false;
	gameEngine.saveSettings(settings);
	resetBoard();
	acceptingClicks = true;
}

void GameView::clearFakeCheckers()
{
	for(uint8_t i = 0; i < fakeItems.size(); i++)
	{
		scene->removeItem(fakeItems[i]);
	}
	fakeItems.clear();
}

void GameView::onCheckerSelected(boardpos_t pos, SquareState checkerType)
{
	// Get moves
	std::vector<Move> moves = gameEngine.getRedMoves(pos);

	// Display moves
	this->drawPossibleMoves(moves, checkerType);
}

void GameView::drawPossibleMoves(std::vector<Move> moves, SquareState checkerType)
{
	// Clear old items
	this->clearFakeCheckers();

	// Add new items
	for(uint8_t i = 0; i < moves.size(); i++)
	{
		FakeCheckerItem* fakeChecker = new FakeCheckerItem(moves[i], checkerType);
		scene->addItem(fakeChecker);
		connect(fakeChecker, &FakeCheckerItem::fakeCheckerSelected, this, &GameView::startRedMove);
		fakeItems.push_back(fakeChecker);
	}
	this->acceptingClicks = true;
}

void GameView::startRedMove(Move move)
{
	this->acceptingClicks = false;
	this->clearFakeCheckers();
	gameEngine.executeRedMove(move);
}

void GameView::blackMoveFinished()
{
	this->acceptingClicks = true;
}

void GameView::displayMove(Move move, bool kingPiece)
{
	CheckerItem* checker = checkers[move.oldPos];
	checker->move(move.newPos);
	checkers[move.newPos] = checker;
	checkers[move.oldPos] = nullptr;
	if(MOVE_ISJUMP(move))
	{
		scene->removeItem(checkers[move.jumpPos]);
		checkers[move.jumpPos] = nullptr;
	}
	if(kingPiece) checker->king();
}

void GameView::resetBoard()
{
	acceptingClicks = false;
	gameEngine.resetGame();
	for(boardpos_t k = 0; k < SQUARE_COUNT; k++)
	{
		SquareState state = gameEngine.getSquareState(k);
		CheckerItem* checker = checkers[k];
		if(checker != nullptr)
		{
			scene->removeItem(checker);
		}
		if(SQUARE_ISEMPTY(state))
		{
			checkers[k] = nullptr;
		}
		else
		{
			checker = new CheckerItem(k, state);
			if(!(SQUARE_ISBLACK(state)))
			{
				connect(checker, &CheckerItem::checkerSelected, this, &GameView::onCheckerSelected);
			}
			scene->addItem(checker);
			checkers[k] = checker;
		}
	}
	acceptingClicks = true;
}

