#ifndef SETTINGSDIALOG_H
#define SETTINGSDIALOG_H

#include <QDialog>
#include "defines.h"

namespace Ui {
class SettingsDialog;
}

class SettingsDialog : public QDialog
{
	Q_OBJECT

public:
	explicit SettingsDialog(QWidget *parent = nullptr);
	~SettingsDialog();

signals:
	void saveSettings(GameSettings settings);

private slots:
	void handleStartGamePress();

private:
	Ui::SettingsDialog *ui;
};

#endif // SETTINGSDIALOG_H
