# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MyDialog.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(456, 338)
        Dialog.setModal(False)
        self.formLayout = QtWidgets.QFormLayout(Dialog)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label)
        self.initSpinBox = QtWidgets.QSpinBox(Dialog)
        self.initSpinBox.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.initSpinBox.sizePolicy().hasHeightForWidth())
        self.initSpinBox.setSizePolicy(sizePolicy)
        self.initSpinBox.setWrapping(False)
        self.initSpinBox.setFrame(True)
        self.initSpinBox.setSuffix("")
        self.initSpinBox.setPrefix("")
        self.initSpinBox.setMinimum(50)
        self.initSpinBox.setMaximum(20000)
        self.initSpinBox.setSingleStep(500)
        self.initSpinBox.setProperty("value", 5000)
        self.initSpinBox.setObjectName("initSpinBox")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.initSpinBox)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.clusterSpinBox = QtWidgets.QSpinBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clusterSpinBox.sizePolicy().hasHeightForWidth())
        self.clusterSpinBox.setSizePolicy(sizePolicy)
        self.clusterSpinBox.setMinimum(10)
        self.clusterSpinBox.setMaximum(3500)
        self.clusterSpinBox.setSingleStep(50)
        self.clusterSpinBox.setProperty("value", 500)
        self.clusterSpinBox.setObjectName("clusterSpinBox")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.clusterSpinBox)
        self.patchSpinBox = QtWidgets.QSpinBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(150)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.patchSpinBox.sizePolicy().hasHeightForWidth())
        self.patchSpinBox.setSizePolicy(sizePolicy)
        self.patchSpinBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.patchSpinBox.setAutoFillBackground(False)
        self.patchSpinBox.setMinimum(3)
        self.patchSpinBox.setMaximum(27)
        self.patchSpinBox.setSingleStep(2)
        self.patchSpinBox.setProperty("value", 3)
        self.patchSpinBox.setObjectName("patchSpinBox")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.patchSpinBox)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_3)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.formLayout.setItem(6, QtWidgets.QFormLayout.LabelRole, spacerItem)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Number of initial patches"))
        self.label_2.setText(_translate("Dialog", "Number of clusters"))
        self.label_3.setText(_translate("Dialog", "Patch Size"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

