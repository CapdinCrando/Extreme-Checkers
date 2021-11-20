#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <QPushButton>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	gameOverBox = new QMessageBox(this);
	gameOverBox->addButton("Yes", QMessageBox::YesRole);
	gameOverBox->addButton("No", QMessageBox::NoRole);
	QPushButton* settingsButton = gameOverBox->addButton("Change Settings", QMessageBox::HelpRole);
	gameOverBox->setModal(true);

	connect(gameOverBox, &QMessageBox::accepted, ui->graphicsView, &GameView::resetBoard);
	connect(gameOverBox, &QMessageBox::rejected, this, &MainWindow::close);
	connect(ui->graphicsView, &GameView::gameOver, this, &MainWindow::gameOver);
	connect(ui->graphicsView, &GameView::printError, this, &MainWindow::displayError);

	settingsDialog = new SettingsDialog();
	connect(settingsDialog, &SettingsDialog::saveSettings, ui->graphicsView, &GameView::saveSettings);
	connect(settingsDialog, &SettingsDialog::rejected, this, &MainWindow::close);
	connect(settingsButton, &QPushButton::clicked, settingsDialog, &SettingsDialog::show);

	settingsDialog->show();
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

void MainWindow::displayError(const std::exception& e)
{
	QMessageBox messageBox;
	messageBox.critical(0, "Error", e.what());
	messageBox.setFixedSize(500,200);
	connect(&messageBox, &QMessageBox::destroyed, this, &MainWindow::close);
}

