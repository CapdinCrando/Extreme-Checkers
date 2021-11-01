#include "settingsdialog.h"
#include "ui_settingsdialog.h"
#include "ai/aimanager.h"

SettingsDialog::SettingsDialog(QWidget *parent) :
	QDialog(parent),
	ui(new Ui::SettingsDialog)
{
	ui->setupUi(this);

	ui->aiSelectionBox->addItems(AIManager::getDescriptionList());

	connect(ui->startGameButton, &QPushButton::clicked, this, &SettingsDialog::handleStartGamePress);
	connect(ui->quitButton, &QPushButton::clicked, this, &SettingsDialog::reject);
}

void SettingsDialog::handleStartGamePress()
{
	GameSettings settings;
	settings.aiLevel = ui->aiSelectionBox->currentIndex();
	emit this->saveSettings(settings);
	this->accept();
}

SettingsDialog::~SettingsDialog()
{
	delete ui;
}
