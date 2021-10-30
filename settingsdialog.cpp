#include "settingsdialog.h"
#include "ui_settingsdialog.h"

SettingsDialog::SettingsDialog(QWidget *parent) :
	QDialog(parent),
	ui(new Ui::SettingsDialog)
{
	ui->setupUi(this);

	ui->aiSelectionBox->addItems(aiModes);

	connect(ui->startGameButton, &QPushButton::clicked, this, &SettingsDialog::handleStartGamePress);
	connect(ui->quitButton, &QPushButton::clicked, this, &SettingsDialog::rejected);
}

void SettingsDialog::handleStartGamePress()
{
	GameSettings settings;
	settings.aiLevel = ui->aiSelectionBox->currentIndex();
	this->saveSettings(settings);
	this->accept();
}

SettingsDialog::~SettingsDialog()
{
	delete ui;
}
