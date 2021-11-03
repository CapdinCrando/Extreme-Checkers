QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11
# remove possible other optimization flags
QMAKE_CXXFLAGS_RELEASE -= -O
QMAKE_CXXFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE -= -Os

# add the desired -O3 if not present
QMAKE_CXXFLAGS_RELEASE *= -O3

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
	ai/aigpu.cpp \
    ai/aimanager.cpp \
    ai/aiminimax.cpp \
    ai/aiparallel.cpp \
    ai/airandom.cpp \
    ai/aitask.cpp \
    ai/aiutility.cpp \
	ai/node.cpp \
    checkeritem.cpp \
    fakecheckeritem.cpp \
    gameboard.cpp \
    gameengine.cpp \
    gameview.cpp \
    main.cpp \
    mainwindow.cpp \
    settingsdialog.cpp

HEADERS += \
    ai/ai.h \
	ai/aigpu.h \
    ai/aimanager.h \
    ai/aiminimax.h \
    ai/aiparallel.h \
    ai/airandom.h \
    ai/aitask.h \
    ai/aiutility.h \
    ai/node.h \
    checkeritem.h \
    defines.h \
    fakecheckeritem.h \
    gameboard.h \
    gameengine.h \
    gameview.h \
    mainwindow.h \
    settingsdialog.h

FORMS += \
    mainwindow.ui \
    settingsdialog.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
	images.qrc
