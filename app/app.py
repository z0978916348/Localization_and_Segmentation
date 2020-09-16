# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap

from main import *

import sys
from os.path import splitext, exists, join


class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 600)
        
        self.img_path = None
        self.models_path = "PyTorch_YOLOv3/checkpoints"
        self.models = list()
        for model in os.listdir("PyTorch_YOLOv3/checkpoints"):
            text = os.path.splitext(model)
            if text[1] == ".pth":
                self.models.append(text[0])
        self.models = sorted(self.models)
        self.current_model = None
        self.Dice_list = None 
        self.bone_num = None

        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 440, 121, 51))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.GetImage)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(250, 440, 121, 51))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.StartSegmentation)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(40, 140, 161, 271))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 140, 161, 271))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(80, 50, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(250, 50, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(410, 140, 161, 271))
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(600, 140, 161, 271))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(450, 50, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")

        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(620, 50, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        

        self.model_label = QtWidgets.QLabel(self.centralwidget)
        self.model_label.setGeometry(QtCore.QRect(420, 450, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.model_label.setFont(font)
        self.model_label.setAlignment(QtCore.Qt.AlignCenter)
        self.model_label.setObjectName("model_label")

        self.model_select = QtWidgets.QComboBox(self.centralwidget)
        self.model_select.setGeometry(QtCore.QRect(520, 450, 121, 31))
        self.model_select.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.model_select.setFixedWidth(200)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(75)
        self.model_select.setFont(font)
        for model in self.models:
            self.model_select.addItem(model)
        self.model_select.currentTextChanged.connect(self.choose_model)
        index = self.model_select.findText("yolov3_ckpt_best_f01")
        self.model_select.setCurrentIndex(index)


        # self.result = QtWidgets.QLabel(self.centralwidget)
        # self.result.setGeometry(QtCore.QRect(520, 500, 121, 31))
        # font = QtGui.QFont()
        # font.setFamily("Bahnschrift SemiLight")
        # font.setPointSize(13)
        # font.setBold(True)
        # font.setWeight(75)
        # self.result.setFont(font)
        # self.result.setAlignment(QtCore.Qt.AlignCenter)
        # self.result.setObjectName("result")
    

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(790, 50, 62, 391))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 4, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.llabel_1 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_1.setObjectName("llabel_1")
        self.verticalLayout_2.addWidget(self.llabel_1)
        self.llabel_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_2.setObjectName("llabel_2")
        self.verticalLayout_2.addWidget(self.llabel_2)
        self.llabel_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_3.setObjectName("llabel_3")
        self.verticalLayout_2.addWidget(self.llabel_3)
        self.llabel_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_4.setObjectName("llabel_4")
        self.verticalLayout_2.addWidget(self.llabel_4)
        self.llabel_5 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_5.setObjectName("llabel_5")
        self.verticalLayout_2.addWidget(self.llabel_5)
        self.llabel_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_6.setObjectName("llabel_6")
        self.verticalLayout_2.addWidget(self.llabel_6)
        self.llabel_7 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_7.setObjectName("llabel_7")
        self.verticalLayout_2.addWidget(self.llabel_7)
        self.llabel_8 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_8.setObjectName("llabel_8")
        self.verticalLayout_2.addWidget(self.llabel_8)
        self.llabel_9 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_9.setObjectName("llabel_9")
        self.verticalLayout_2.addWidget(self.llabel_9)
        self.llabel_10 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.llabel_10.setObjectName("llabel_10")
        self.verticalLayout_2.addWidget(self.llabel_10)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(1000, 50, 62, 391))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 4, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.llabel_11 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_11.setObjectName("llabel_11")
        self.verticalLayout_3.addWidget(self.llabel_11)
        self.llabel_12 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_12.setObjectName("llabel_12")
        self.verticalLayout_3.addWidget(self.llabel_12)
        self.llabel_13 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_13.setObjectName("llabel_13")
        self.verticalLayout_3.addWidget(self.llabel_13)
        self.llabel_14 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_14.setObjectName("llabel_14")
        self.verticalLayout_3.addWidget(self.llabel_14)
        self.llabel_15 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_15.setObjectName("llabel_15")
        self.verticalLayout_3.addWidget(self.llabel_15)
        self.llabel_16 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_16.setObjectName("llabel_16")
        self.verticalLayout_3.addWidget(self.llabel_16)
        self.llabel_17 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_17.setObjectName("llabel_17")
        self.verticalLayout_3.addWidget(self.llabel_17)
        self.llabel_18 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_18.setObjectName("llabel_18")
        self.verticalLayout_3.addWidget(self.llabel_18)
        self.llabel_19 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_19.setObjectName("llabel_19")
        self.verticalLayout_3.addWidget(self.llabel_19)
        self.llabel_20 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.llabel_20.setObjectName("llabel_20")
        self.verticalLayout_3.addWidget(self.llabel_20)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(880, 50, 51, 400))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.lineEdit_1 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.verticalLayout_4.addWidget(self.lineEdit_1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout_4.addWidget(self.lineEdit_2)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.verticalLayout_4.addWidget(self.lineEdit_3)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.verticalLayout_4.addWidget(self.lineEdit_4)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.verticalLayout_4.addWidget(self.lineEdit_5)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.verticalLayout_4.addWidget(self.lineEdit_6)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.verticalLayout_4.addWidget(self.lineEdit_7)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.verticalLayout_4.addWidget(self.lineEdit_8)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.verticalLayout_4.addWidget(self.lineEdit_9)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.verticalLayout_4.addWidget(self.lineEdit_10)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(1090, 50, 51, 400))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.lineEdit_11 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.verticalLayout_5.addWidget(self.lineEdit_11)
        self.lineEdit_12 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.verticalLayout_5.addWidget(self.lineEdit_12)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.verticalLayout_5.addWidget(self.lineEdit_13)
        self.lineEdit_14 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.verticalLayout_5.addWidget(self.lineEdit_14)
        self.lineEdit_15 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.verticalLayout_5.addWidget(self.lineEdit_15)
        self.lineEdit_16 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.verticalLayout_5.addWidget(self.lineEdit_16)
        self.lineEdit_17 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.verticalLayout_5.addWidget(self.lineEdit_17)
        self.lineEdit_18 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.verticalLayout_5.addWidget(self.lineEdit_18)
        self.lineEdit_19 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_19.setObjectName("lineEdit_19")
        self.verticalLayout_5.addWidget(self.lineEdit_19)
        self.lineEdit_20 = QtWidgets.QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_20.setObjectName("lineEdit_20")
        self.verticalLayout_5.addWidget(self.lineEdit_20)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.average = QtWidgets.QLabel(self.centralwidget)
        self.average.setGeometry(QtCore.QRect(750, 500, 150, 30))
        self.average.setAlignment(QtCore.Qt.AlignCenter)
        self.average.setObjectName("average")
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
    
        self.lineEdit_average = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_average.setGeometry(QtCore.QRect(880, 500, 70, 30))
        self.lineEdit_average.setObjectName("lineEdit_average")

        self.num = QtWidgets.QLabel(self.centralwidget)
        self.num.setGeometry(QtCore.QRect(950, 530, 150, 30))
        self.num.setAlignment(QtCore.Qt.AlignCenter)
        self.num.setObjectName("num")
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_num = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_num.setGeometry(QtCore.QRect(1080, 530, 70, 30))
        self.lineEdit_num.setObjectName("lineEdit_num")

        self.original = QtWidgets.QLabel(self.centralwidget)
        self.original.setGeometry(QtCore.QRect(950, 500, 150, 30))
        self.original.setAlignment(QtCore.Qt.AlignCenter)
        self.original.setObjectName("original")
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiLight")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_original = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_original.setGeometry(QtCore.QRect(1080, 500, 70, 30))
        self.lineEdit_original.setObjectName("lineEdit_orginal")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # self.result.setText("0.99")



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Strart"))
        self.label.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", ""))
        self.label_3.setText(_translate("MainWindow", "Input"))
        self.label_4.setText(_translate("MainWindow", "Localization"))
        self.label_5.setText(_translate("MainWindow", ""))
        self.label_6.setText(_translate("MainWindow", ""))
        self.label_7.setText(_translate("MainWindow", "Predict"))
        self.label_8.setText(_translate("MainWindow", "GT"))

        self.model_label.setText(_translate("MainWindow", "Model :"))

        self.llabel_1.setText(_translate("MainWindow", "bone 1"))
        self.llabel_2.setText(_translate("MainWindow", "bone 2"))
        self.llabel_3.setText(_translate("MainWindow", "bone 3"))
        self.llabel_4.setText(_translate("MainWindow", "bone 4"))
        self.llabel_5.setText(_translate("MainWindow", "bone 5"))
        self.llabel_6.setText(_translate("MainWindow", "bone 6"))
        self.llabel_7.setText(_translate("MainWindow", "bone 7"))
        self.llabel_8.setText(_translate("MainWindow", "bone 8"))
        self.llabel_9.setText(_translate("MainWindow", "bone 9"))
        self.llabel_10.setText(_translate("MainWindow", "bone 10"))
        self.llabel_11.setText(_translate("MainWindow", "bone 11"))
        self.llabel_12.setText(_translate("MainWindow", "bone 12"))
        self.llabel_13.setText(_translate("MainWindow", "bone 13"))
        self.llabel_14.setText(_translate("MainWindow", "bone 14"))
        self.llabel_15.setText(_translate("MainWindow", "bone 15"))
        self.llabel_16.setText(_translate("MainWindow", "bone 16"))
        self.llabel_17.setText(_translate("MainWindow", "bone 17"))
        self.llabel_18.setText(_translate("MainWindow", "bone 18"))
        self.llabel_19.setText(_translate("MainWindow", "bone 19"))
        self.llabel_20.setText(_translate("MainWindow", "bone 20"))
        
        self.average.setText(_translate("MainWindow", "Average Dice"))
        self.num.setText(_translate("MainWindow", "num of bones"))
        self.original.setText(_translate("MainWindow", "original bones"))

    def GetImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', r"./source/image", "Image files (*.png)")
        self.label.setPixmap(QPixmap(file_path))
        self.label.setScaledContents(True)
        self.img_path = splitext(file_path)[0].replace("/home/p76094266/source/image/", "")
        self.clear()

    def FindImage(self):
        img = self.img_path[-4:] + ".png" 
        
        localization_path = f"./output/{img}"
        predictImg_path = f"./result/p{img}"
        groundTruth_path = f"./source/label/{img}"

        self.label_2.setPixmap(QPixmap(localization_path))
        self.label_5.setPixmap(QPixmap(predictImg_path))
        self.label_6.setPixmap(QPixmap(groundTruth_path))

        self.label_2.setScaledContents(True)
        self.label_5.setScaledContents(True)
        self.label_6.setScaledContents(True)
        
    def StartSegmentation(self):
        
        # Object_detection
        delete_valid_data(r"./valid_data")
        delete_valid_data(r"./coordinate")
        
        if self.current_model != None:
            detect_one(self.img_path, self.current_model)
        else:
            detect_one(self.img_path)

        # Segmentation
        self.Dice_list, self.bone_num = Segmentation_one(self.img_path[-4:] + ".png" )
        
        self.FindImage()

        self.lineEdit_1.setText(str(self.Dice_list[0]))
        self.lineEdit_2.setText(str(self.Dice_list[1]))
        self.lineEdit_3.setText(str(self.Dice_list[2]))
        self.lineEdit_4.setText(str(self.Dice_list[3]))
        self.lineEdit_5.setText(str(self.Dice_list[4]))
        self.lineEdit_6.setText(str(self.Dice_list[5]))
        self.lineEdit_7.setText(str(self.Dice_list[6]))
        self.lineEdit_8.setText(str(self.Dice_list[7]))
        self.lineEdit_9.setText(str(self.Dice_list[8]))
        self.lineEdit_10.setText(str(self.Dice_list[9]))
        self.lineEdit_11.setText(str(self.Dice_list[10]))
        self.lineEdit_12.setText(str(self.Dice_list[11]))
        self.lineEdit_13.setText(str(self.Dice_list[12]))
        self.lineEdit_14.setText(str(self.Dice_list[13]))
        self.lineEdit_15.setText(str(self.Dice_list[14]))
        self.lineEdit_16.setText(str(self.Dice_list[15]))
        self.lineEdit_17.setText(str(self.Dice_list[16]))
        self.lineEdit_18.setText(str(self.Dice_list[17]))
        self.lineEdit_19.setText(str(self.Dice_list[18]))
        self.lineEdit_20.setText(str(self.Dice_list[19]))
        
        label_path = "source/label/" + self.img_path[-4:] + ".png"
        num = self.connected_component_label(label_path)
        sum = 0
        for i in range(num):
            sum += self.Dice_list[i]
        print(sum)
        sum /= self.bone_num
        
        average = round(sum, 2)
        self.lineEdit_average.setText(str(average))
        
        
        
        
        self.lineEdit_original.setText(str(num))
        self.lineEdit_num.setText(str(self.bone_num))
        
    def choose_model(self, name):
        self.current_model = os.path.join(self.models_path, f"{name}.pth")
        print(f"Change Model to {self.current_model}")

    def clear(self):
        self.lineEdit_1.setText(str(0))
        self.lineEdit_2.setText(str(0))
        self.lineEdit_3.setText(str(0))
        self.lineEdit_4.setText(str(0))
        self.lineEdit_5.setText(str(0))
        self.lineEdit_6.setText(str(0))
        self.lineEdit_7.setText(str(0))
        self.lineEdit_8.setText(str(0))
        self.lineEdit_9.setText(str(0))
        self.lineEdit_10.setText(str(0))
        self.lineEdit_11.setText(str(0))
        self.lineEdit_12.setText(str(0))
        self.lineEdit_13.setText(str(0))
        self.lineEdit_14.setText(str(0))
        self.lineEdit_15.setText(str(0))
        self.lineEdit_16.setText(str(0))
        self.lineEdit_17.setText(str(0))
        self.lineEdit_18.setText(str(0))
        self.lineEdit_19.setText(str(0))
        self.lineEdit_20.setText(str(0))

        self.lineEdit_average.setText(str(0))
        self.lineEdit_num.setText(str(0))
        self.lineEdit_original.setText(str(0))

    def connected_component_label(self, path):
    
        # Getting the input image
        img = cv2.imread(path, 0)
        
        # Converting those pixels with values 1-127 to 0 and others to 1
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        # Applying cv2.connectedComponents() 
        num_labels, labels = cv2.connectedComponents(img)
        return num_labels - 1


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

