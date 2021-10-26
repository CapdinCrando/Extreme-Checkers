#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	gameOverBox = new QMessageBox(this);
	gameOverBox->addButton("Yes", QMessageBox::YesRole);
	gameOverBox->addButton("No", QMessageBox::NoRole);
	gameOverBox->setModal(true);
	connect(gameOverBox, &QMessageBox::accepted, ui->graphicsView, &GameView::resetBoard);
	connect(gameOverBox, &QMessageBox::rejected, this, &MainWindow::close);
	connect(ui->graphicsView, &GameView::gameOver, this, &MainWindow::gameOver);
}

MainWindow::~MainWindow()
{
	delete gameOverBox;
    delete ui;
}

void MainWindow::gameOver(GameState gameState)
{
	if(gameState == GAME_OVER_RED_WIN) gameOverBox->setText("You won!\n Would you like to play again?");
	else if(gameState == GAME_OVER_BLACK_WIN) gameOverBox->setText("You lost.\n Would you like to play again?");
	else if(gameState == GAME_OVER_TIE) gameOverBox->setText("The game was a tie!\n Would you like to play again?");
	else return;
	gameOverBox->show();
}

