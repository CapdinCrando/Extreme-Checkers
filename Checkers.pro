QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

QMAKE_CXXFLAGS_RELEASE += -O3

SOURCES += \
    ai/aimanager.cpp \
    ai/aiminimax.cpp \
    ai/aiparallel.cpp \
    ai/airandom.cpp \
    ai/aitask.cpp \
    ai/aiutility.cpp \
    checkeritem.cpp \
    fakecheckeritem.cpp \
    gameboard.cpp \
    gameengine.cpp \
	gamelogger.cpp \
    gameview.cpp \
    main.cpp \
    mainwindow.cpp \
    settingsdialog.cpp \
	transposition/table.cpp \
	transposition/tableentry.cpp

HEADERS += \
	ai/ai.h \
    ai/aimanager.h \
    ai/aiminimax.h \
    ai/aiparallel.h \
    ai/airandom.h \
    ai/aitask.h \
    ai/aiutility.h \
	ai/gpuutility.h \
    checkeritem.h \
    defines.h \
    fakecheckeritem.h \
    gameboard.h \
    gameengine.h \
	gamelogger.h \
    gameview.h \
    mainwindow.h \
    settingsdialog.h \
	transposition/table.h \
	transposition/tableentry.h

FORMS += \
    mainwindow.ui \
    settingsdialog.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
	images.qrc
