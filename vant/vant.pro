#-------------------------------------------------
#
# Project created by QtCreator 2016-11-16T15:44:29
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = vant
TEMPLATE = app

LIBS += `pkg-config \
    opencv \
    --cflags \
    --libs`

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    rgb_hist.h

FORMS    += mainwindow.ui
