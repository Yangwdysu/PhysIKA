/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the demonstration applications of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/
#include "PMainWindow.h"
#include "PDockWidget.h"
#include "PToolBar.h"
#include "PStatusBar.h"
#include "PVTKOpenGLWidget.h"
#include "PAnimationWidget.h"

#include "PIODockWidget.h"
#include "PLogWidget.h"
#include "PConsoleWidget.h"
#include "PSceneGraphWidget.h"
#include "PPropertyWidget.h"'
#include "PModuleListWidget.h"

#include <QAction>
#include <QLayout>
#include <QMenu>
#include <QMenuBar>
#include <QStatusBar>
#include <QTextEdit>
#include <QFile>
#include <QDataStream>
#include <QFileDialog>
#include <QDialogButtonBox>
#include <QMessageBox>
#include <QSignalMapper>
#include <QApplication>
#include <QPainter>
#include <QMouseEvent>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QDebug>
#include <QtWidgets/QOpenGLWidget>

// #include "Node/NodeData.hpp"
// #include "Node/FlowScene.hpp"
// #include "Node/FlowView.hpp"
// #include "Node/FlowViewStyle.hpp"
// #include "Node/ConnectionStyle.hpp"
// #include "Node/DataModelRegistry.hpp"

//#include "models.h"

namespace PhysIKA
{
	Q_DECLARE_METATYPE(QDockWidget::DockWidgetFeatures)

	PMainWindow::PMainWindow(QWidget *parent, Qt::WindowFlags flags)
		: QMainWindow(parent, flags),
		m_statusBar(nullptr),
		m_vtkOpenglWidget(nullptr),
		m_scenegraphWidget(nullptr),
		m_propertyWidget(nullptr),
		m_animationWidget(nullptr),
		m_moduleListWidget(nullptr)
	{
		setObjectName("MainWindow");
		setWindowTitle("PhysIKA Studio");

		setCentralView();
		setupToolBar();
		setupStatusBar();
		setupMenuBar();
		setupAllWidgets();

		statusBar()->showMessage(tr("Status Bar"));
	}

	void PMainWindow::mainLoop()
	{
	}

	void PMainWindow::createWindow(int width, int height)
	{

	}

	void PMainWindow::newScene()
	{
		QMessageBox::StandardButton reply;

		reply = QMessageBox::question(this, "Save", "Do you want to save your changes?",
			QMessageBox::Ok | QMessageBox::Cancel);
	}

	void PMainWindow::setCentralView()
	{
		QWidget* centralWidget = new QWidget();
		setCentralWidget(centralWidget);

		centralWidget->setContentsMargins(0, 0, 0, 0);
		QGridLayout* mainLayout = new QGridLayout();
		mainLayout->setContentsMargins(0, 0, 0, 0);
		mainLayout->setSpacing(0);
		centralWidget->setLayout(mainLayout);


		//Setup views
		QTabWidget* tabWidget = new QTabWidget();
		tabWidget->setObjectName(QStringLiteral("tabWidget"));
		tabWidget->setGeometry(QRect(140, 60, 361, 241));

		m_vtkOpenglWidget = new PVTKOpenGLWidget();
		m_vtkOpenglWidget->setObjectName(QStringLiteral("tabView"));
		m_vtkOpenglWidget->layout()->setMargin(0);
		tabWidget->addTab(m_vtkOpenglWidget, QString());

		QWidget* tabEditor = new QWidget();
		tabEditor->setObjectName(QStringLiteral("tabEditor"));
		tabWidget->addTab(tabEditor, QString());

		tabWidget->setTabText(tabWidget->indexOf(m_vtkOpenglWidget), QApplication::translate("MainWindow", "View", Q_NULLPTR));
		tabWidget->setTabText(tabWidget->indexOf(tabEditor), QApplication::translate("MainWindow", "Node Editor", Q_NULLPTR));


		//Setup animation widget
		m_animationWidget = new PAnimationWidget(this);
		m_animationWidget->layout()->setMargin(0);

 		mainLayout->addWidget(tabWidget, 0, 0);
 		mainLayout->addWidget(m_animationWidget, 1, 0);
	}

	void PMainWindow::setupToolBar()
	{
		PToolBar *tb = new PToolBar(tr("Tool Bar"), this);
		toolBars.append(tb);
		addToolBar(tb);
	}

	void PMainWindow::setupStatusBar()
	{
		m_statusBar = new PStatusBar(this);
		setStatusBar(m_statusBar);
	}

	void PMainWindow::setupMenuBar()
	{
		QMenu *menu = menuBar()->addMenu(tr("&File"));

		menu->addAction(tr("New ..."), this, &PMainWindow::newScene);
		menu->addAction(tr("Load ..."), this, &PMainWindow::loadScene);
		menu->addAction(tr("Save ..."), this, &PMainWindow::saveScene);

		menu->addSeparator();
		menu->addAction(tr("&Quit"), this, &QWidget::close);

		mainWindowMenu = menuBar()->addMenu(tr("&View"));
		mainWindowMenu->addAction(tr("FullScreen"), this, &PMainWindow::fullScreen);

#ifdef Q_OS_OSX
		toolBarMenu->addSeparator();

		action = toolBarMenu->addAction(tr("Unified"));
		action->setCheckable(true);
		action->setChecked(unifiedTitleAndToolBarOnMac());
		connect(action, &QAction::toggled, this, &QMainWindow::setUnifiedTitleAndToolBarOnMac);
#endif

		windowMenu = menuBar()->addMenu(tr("&Window"));
		for (int i = 0; i < toolBars.count(); ++i)
			windowMenu->addMenu(toolBars.at(i)->toolbarMenu());

		aboutMenu = menuBar()->addMenu(tr("&Help"));
		aboutMenu->addAction(tr("Show Help ..."), this, &PMainWindow::showHelp);
		aboutMenu->addAction(tr("About ..."), this, &PMainWindow::showAbout);
	}

	void PMainWindow::saveScene()
	{
		return;
	}

	void PMainWindow::fullScreen()
	{
		return;
	}

	void PMainWindow::showHelp()
	{
		return;
	}

	void PMainWindow::showAbout()
	{
		QMessageBox::about(this, tr("PhysLab"), tr("PhysLab 1.0"));
		return;
	}

	void PMainWindow::loadScene()
	{
		return;
	}

	void PMainWindow::setupAllWidgets()
	{
		qRegisterMetaType<QDockWidget::DockWidgetFeatures>();

		windowMenu->addSeparator();

		static const struct Set {
			const char * name;
			uint flags;
			Qt::DockWidgetArea area;
		} sets[] = {
			{ "White", 0, Qt::LeftDockWidgetArea },
			{ "Blue", 0, Qt::BottomDockWidgetArea },
			{ "Yellow", 0, Qt::RightDockWidgetArea }
		};
		const int setCount = sizeof(sets) / sizeof(Set);

		const QIcon qtIcon(QPixmap(":/res/qt.png"));

		PDockWidget *leftDockWidget = new PDockWidget(tr(sets[0].name), this, Qt::WindowFlags(sets[0].flags));
		leftDockWidget->setWindowTitle("Scene Browser");
		leftDockWidget->setWindowIcon(qtIcon);
		addDockWidget(sets[0].area, leftDockWidget);
		windowMenu->addMenu(leftDockWidget->colorSwatchMenu());

		m_scenegraphWidget = new PSceneGraphWidget();
		leftDockWidget->setWidget(m_scenegraphWidget);

		PDockWidget *moduleListDockWidget = new PDockWidget(tr(sets[0].name), this, Qt::WindowFlags(sets[0].flags));
		moduleListDockWidget->setWindowTitle("Module Editor");
		moduleListDockWidget->setWindowIcon(qtIcon);
		addDockWidget(sets[0].area, moduleListDockWidget);
		windowMenu->addMenu(moduleListDockWidget->colorSwatchMenu());

		m_moduleListWidget = new PModuleListWidget();
		moduleListDockWidget->setWidget(m_moduleListWidget);


		PIODockWidget *bottomDockWidget = new PIODockWidget(this, Qt::WindowFlags(sets[1].flags));
		bottomDockWidget->setWindowIcon(qtIcon);
		addDockWidget(sets[1].area, bottomDockWidget);
		windowMenu->addMenu(bottomDockWidget->colorSwatchMenu());


		PDockWidget *rightDockWidget = new PDockWidget(tr(sets[2].name), this, Qt::WindowFlags(sets[2].flags));
		rightDockWidget->setWindowTitle("Property Editor");
		rightDockWidget->setWindowIcon(qtIcon);
		addDockWidget(sets[2].area, rightDockWidget);
		windowMenu->addMenu(rightDockWidget->colorSwatchMenu());

		m_propertyWidget = new PPropertyWidget();
		rightDockWidget->setWidget(m_propertyWidget);

		setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
		setCorner(Qt::BottomRightCorner, Qt::RightDockWidgetArea);

		connect(m_scenegraphWidget, SIGNAL(notifyNodeSelected(Node*)), m_moduleListWidget, SLOT(updateModule(Node*)));
		connect(m_scenegraphWidget, SIGNAL(notifyNodeSelected(Node*)), m_propertyWidget, SLOT(showProperty(Node*)));
		connect(m_moduleListWidget, SIGNAL(notifyModuleSelected(Module*)), m_propertyWidget, SLOT(showProperty(Module*)));
	}

	void PMainWindow::mousePressEvent(QMouseEvent *event)
	{
		// 	QLichtThread* m_thread = new QLichtThread(openGLWidget->winId());
		// 	m_thread->start();
	}

}