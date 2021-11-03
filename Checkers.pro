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
	#ai/gpuutility.cu \
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

# Define output directories
DESTDIR = ../bin
CUDA_OBJECTS_DIR = OBJECTS_DIR/../cuda

# This makes the .cu files appear in your project
# OTHER_FILES += \
#     vectorAdd.cu
CUDA_SOURCES += \
	ai/gpuutility.cu

#OTHER_FILES += ai/gpuutility.cu
SOURCES += ai/gpuutility.cu
SOURCES -= ai/gpuutility.cu

#-------------------------------------------------

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

# CUDA settings
CUDA_DIR = $$(CUDA_PATH)            # Path to cuda toolkit install
SYSTEM_NAME = x64                   # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64                    # '32' or '64', depending on your system
CUDA_ARCH = sm_61                   # Type of CUDA architecture
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $$CUDA_DIR\include \
			   $$CUDA_DIR/common/inc \
			   $$CUDA_DIR/../shared/inc

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME \
				$$CUDA_DIR/common/lib/$$SYSTEM_NAME \
				$$CUDA_DIR/../shared/lib/$$SYSTEM_NAME

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# Add the necessary libraries
CUDA_LIB_NAMES = cudart_static kernel32 user32 gdi32 winspool comdlg32 \
				 advapi32 shell32 ole32 oleaut32 uuid odbc32 odbccp32 \
				 #freeglut glew32

for(lib, CUDA_LIB_NAMES) {
	CUDA_LIBS += -l$$lib
}
LIBS += $$CUDA_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
	# Debug mode
	cuda_d.input = CUDA_SOURCES
	cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
	cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
					  --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
					  --compile -cudart static -g -DWIN32 -D_MBCS \
					  -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
					  -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
					  -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
	cuda_d.dependency_type = TYPE_C
	QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
	# Release mode
	cuda.input = CUDA_SOURCES
	cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
	cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
					--machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
					--compile -cudart static -DWIN32 -D_MBCS \
					-Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
					-Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
					-c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
	cuda.dependency_type = TYPE_C
	QMAKE_EXTRA_COMPILERS += cuda
}
