from skimage.color import rgb2gray
from skimage import io
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from net.predict import predict_one
from utils import clean_noice, horizontal_cut, draw_edge, dice_coef_each_region
import os
import sys


class MainWindows(QtWidgets.QMainWindow):

    def __init__(self, models_path="save_model", temp_dir="temp"):
        super().__init__()
        self.ratio = 0.7
        self.input = None
        self.truth = None
        self.output = None
        self.models_path = models_path
        self.temp_dir = temp_dir
        self.models = list()
        for model in os.listdir(models_path):
            text = os.path.splitext(model)
            if text[1] == ".pt":
                self.models.append(text[0])
        self.models = sorted(self.models)
        self.current_model = os.path.join(models_path, f"{self.models[0]}.pt")
        self.initUI()

    def initUI(self):
        # mainwindows
        self.resize(1200, 600)
        self.setFont(QtGui.QFont("微軟正黑體", 12))
        self.center()
        self.setWindowTitle("Vertebra Segmentation")

        # widget
        self.widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.widget)

        hbox = QtWidgets.QHBoxLayout()
        self.widget.setLayout(hbox)

        v1 = QtWidgets.QVBoxLayout()
        v1.setAlignment(QtCore.Qt.AlignHCenter)
        label_origin = QtWidgets.QLabel("原始圖片", self)
        label_origin.setAlignment(QtCore.Qt.AlignHCenter)
        self.picture_origin = QtWidgets.QLabel("Picture", self)
        v1.addWidget(label_origin)
        v1.addWidget(self.picture_origin)
        hbox.addLayout(v1)

        v2 = QtWidgets.QVBoxLayout()
        v2.setAlignment(QtCore.Qt.AlignHCenter)
        label_truth = QtWidgets.QLabel("Ground Truth", self)
        label_truth.setAlignment(QtCore.Qt.AlignHCenter)
        self.picture_truth = QtWidgets.QLabel("Picture", self)
        v2.addWidget(label_truth)
        v2.addWidget(self.picture_truth)
        hbox.addLayout(v2)

        v3 = QtWidgets.QVBoxLayout()
        v3.setAlignment(QtCore.Qt.AlignHCenter)
        label_predict = QtWidgets.QLabel("預測圖片", self)
        label_predict.setAlignment(QtCore.Qt.AlignHCenter)
        self.picture_predict = QtWidgets.QLabel("Picture", self)
        v3.addWidget(label_predict)
        v3.addWidget(self.picture_predict)
        hbox.addLayout(v3)

        v4 = QtWidgets.QVBoxLayout()
        v4.setAlignment(QtCore.Qt.AlignHCenter)
        model_layout = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel("Model:", self)
        model_select = QtWidgets.QComboBox(self)
        model_select.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        model_select.setFixedWidth(150)
        for model in self.models:
            model_select.addItem(model)
        model_select.currentTextChanged.connect(self.choose_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_select)
        ratio_layout = QtWidgets.QHBoxLayout()
        ratio_label = QtWidgets.QLabel("Ratio:", self)
        self.ratio_textbox = QtWidgets.QLineEdit(str(self.ratio))
        self.ratio_textbox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.ratio_textbox.setFixedWidth(150)
        self.ratio_textbox.returnPressed.connect(self.change_ratio)
        ratio_layout.addWidget(ratio_label)
        ratio_layout.addWidget(self.ratio_textbox)
        dice_label = QtWidgets.QLabel("Dice", self)
        dice_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.dice = QtWidgets.QLabel("", self)
        self.dice.setAlignment(QtCore.Qt.AlignHCenter)
        v4.addLayout(model_layout)
        v4.addLayout(ratio_layout)
        v4.addWidget(dice_label)
        v4.addWidget(self.dice)
        hbox.addLayout(v4)

        # toolbar
        self.open_image = QtWidgets.QAction("開啟原始圖片", self)
        self.open_image.triggered.connect(self.load_input)
        self.open_truth = QtWidgets.QAction("開啟Groud Truth", self)
        self.open_truth.triggered.connect(self.load_truth)
        self.predict_button = QtWidgets.QAction("預測", self)
        self.predict_button.triggered.connect(self.predict_now)
        self.calculate = QtWidgets.QAction("計算Dice", self)
        self.calculate.triggered.connect(self.calculate_dice)

        self.toolbar = self.addToolBar("Toolbar")
        self.toolbar.addAction(self.open_image)
        self.toolbar.addAction(self.open_truth)
        self.toolbar.addAction(self.predict_button)
        self.toolbar.addAction(self.calculate)

        self.show()

    def calculate_dice(self):
        if self.output is not None and self.truth is not None:
            if self.output.shape != self.truth.shape:
                self.truth = self.truth[:self.output.shape[0], :self.output.shape[1]]
            scores, mean = dice_coef_each_region(self.output, self.truth)
            blue = "blue"
            red = "red"
            space = "&nbsp;"
            dices = [f"{i}:{space * 5}<font color={red if k < 0.7 else blue}>{round(k * 1000) / 10}%</font>" for i, j, k in scores]
            dices += ["<br>", f"Avg:{space * 5}<font color={red if mean < 0.7 else blue}>{round(mean * 1000) / 10}%</font>"]
            dice_msg = "<br>".join(dices)
            self.dice.setText(dice_msg + "<br>" * 5)

    def predict_now(self):
        if self.input is not None:
            print(f"Predict Mask")
            self.dice.setText("")
            output = predict_one(self.input, self.current_model)
            output = clean_noice(output)
            output = horizontal_cut(output, ratio=self.ratio)
            self.output = output
            self.show_predict(draw_edge(self.input, output))
            print(f"Predict Done")

    def choose_model(self, name):
        self.current_model = os.path.join(self.models_path, f"{name}.pt")
        print(f"Change Model to {self.current_model}")

    def change_ratio(self):
        try:
            self.ratio = float(self.ratio_textbox.text())
            print(f"Change Ratio to {self.ratio}")
        except Exception:
            self.ratio_textbox.setText(str(self.ratio))

    def center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)

    def get_file(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", ".", "Image files (*.jpg *.png)")
        return fname

    def load_input(self):
        path = self.get_file()[0]
        if path:
            self.dice.setText("")
            self.input = rgb2gray(io.imread(path))
            img = QtGui.QImage(self.input, self.input.shape[1], self.input.shape[0], QtGui.QImage.Format_Grayscale8)
            img = img.scaled(250, 600)
            pixmap = QtGui.QPixmap(img)
            self.picture_origin.setPixmap(pixmap)
            self.picture_origin.show()

    def load_truth(self):
        path = self.get_file()[0]
        if path:
            self.dice.setText("")
            self.truth = rgb2gray(io.imread(path))
            img = QtGui.QImage(self.truth, self.truth.shape[1], self.truth.shape[0], QtGui.QImage.Format_Grayscale8)
            img = img.scaled(250, 600)
            pixmap = QtGui.QPixmap(img)
            self.picture_truth.setPixmap(pixmap)
            self.picture_truth.show()

    def show_predict(self, img, save_file="predict_temp.png"):
        path = os.path.join(self.temp_dir, save_file)
        io.imsave(path, img)
        img = QtGui.QImage(io.imread(path), img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        img = img.scaled(250, 600)
        pixmap = QtGui.QPixmap(img)
        self.picture_predict.setPixmap(pixmap)
        self.picture_predict.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    windows = MainWindows()
    sys.exit(app.exec_())
