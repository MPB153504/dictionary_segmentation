from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from UI.MainWindow import Ui_MainWindow
from UI.MyDialog import Ui_Dialog
import types

import numpy as np
from dictionary_segmentation import dict_seg as dsm

import qimage2ndarray

BRUSH_INI_SIZE = 20


COLORS = [
    '#C10000', '#42A077', '#823783', '#32A7AB', '#CDCB41', '#D491B3', '#3370AA',
    '#E39627', '#7B0040', '#3F310E', '#20FC9B', '#BDFFC6', '#A38184', '#B9A5BF',
]

MODES = ['eraser', 'brush']


class GraphView(QGraphicsView):

    primary_color = QColor(Qt.black)
    primary_color_updated = pyqtSignal(str)

    active_color = None

    # size and opacity of brush
    config = {
        'size': 1,
        'opacity': 0.5
    }

    def initialize(self):

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.image_is_uploaded = False

        self.set_mode('brush')

    def loadImage(self, image):

        self.image = image
        self.imlayer = QGraphicsPixmapItem(self.image)
        self.imlayer.setZValue(1)
        self.addlayer(self.imlayer)

        pog = QPixmap(self.image.size())
        pog.fill(Qt.transparent)
        self.roilayer = QGraphicsPixmapItem(pog)
        self.roilayer.setZValue(10)
        self.roilayer.setOpacity(self.config['opacity'])
        self.addlayer(self.roilayer)

    def loadLabels(self, image):

        self.image = image
        self.roilayer.setPixmap(self.image)

    def update_opacity(self):
        self.roilayer.setOpacity(self.config['opacity'])

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.updateviewer()
        self.set_brush_cursor()

    def scaleImage(self, factor):
        self.scale(factor, factor)
        self.set_brush_cursor()

    def addlayer(self, layer):
        self.scene.addItem(layer)
        self.updateviewer()

    def updateviewer(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatioByExpanding)

    def set_primary_color(self, hex):
        self.primary_color = QColor(hex)

    def set_config(self, key, value):
        self.config[key] = value

    def set_mode(self, mode):
        self.last_pos = None
        self.mode = mode

    def reset_mode(self):
        self.set_mode(self.mode)

    # Mouse events #

    def mousePressEvent(self, e):
        fn = getattr(self, "%s_mousePressEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseMoveEvent(self, e):
        fn = getattr(self, "%s_mouseMoveEvent" % self.mode, None)
        if (fn and self.image_is_uploaded):
            return fn(e)

    def mouseReleaseEvent(self, e):
        fn = getattr(self, "%s_mouseReleaseEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseDoubleClickEvent(self, e):
        fn = getattr(self, "%s_mouseDoubleClickEvent" % self.mode, None)
        if fn:
            return fn(e)

    # Generic events (shared by brush-like tools)

    def generic_mousePressEvent(self, e):
        self.last_pos = self.mapToScene(e.pos())

        if e.button() == Qt.LeftButton:
            self.active_color = self.primary_color

    def generic_mouseReleaseEvent(self, e):
        self.last_pos = None

    # Mode-specific events.

    # Eraser events

    def eraser_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)

    def eraser_mouseMoveEvent(self, e):
        if self.last_pos:

            new_position = self.mapToScene(e.pos())

            pixmap = self.roilayer.pixmap()
            painter = QPainter(pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setPen(QPen(QColor(Qt.white), self.config['size'], Qt.SolidLine,Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_pos, new_position)
            self.roilayer.setPixmap(pixmap)
            painter.end()

            self.modified = True

            self.update()
            self.last_pos = new_position

    def eraser_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

    # Brush events

    def brush_mousePressEvent(self, e):
        self.generic_mousePressEvent(e)

    def brush_mouseMoveEvent(self, e):

        if self.last_pos:

            pf = self.mapToScene(e.pos())

            pixmap = self.roilayer.pixmap()
            painter = QPainter(pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setPen(QPen(self.active_color, self.config['size'], Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_pos, pf)

            self.roilayer.setPixmap(pixmap)
            painter.end()

            self.modified = True

            self.update()

            self.last_pos = pf

    def brush_mouseReleaseEvent(self, e):
        self.generic_mouseReleaseEvent(e)

        if self.autoRunBtn.isChecked():
            self.run_results()

    def set_brush_cursor(self):

        transform = self.transform()

        x_scale = transform.m11()
        y_scale = transform.m22()

        brush_size_x = self.config['size']*x_scale
        brush_size_y = self.config['size']*y_scale

        if self.mode == 'brush':

            cursor_pixmap = QPixmap(brush_size_x, brush_size_y)
            cursor_pixmap.fill(Qt.transparent)  # Otherwise you get a black background
            painter_cursor = QPainter(cursor_pixmap)

            brush_color = self.primary_color

            painter_cursor.setPen(Qt.NoPen)  # Otherwise you get an thin black border
            painter_cursor.setBrush(brush_color)
            painter_cursor.setOpacity(self.config['opacity'])

            painter_cursor.drawEllipse(0, 0, brush_size_x, brush_size_y)
            painter_cursor.end()

            m_Cursor = QCursor(cursor_pixmap)
            self.setCursor(m_Cursor)

        elif self.mode == 'eraser':

            cursor_pixmap = QPixmap(brush_size_x+1, brush_size_y+1)
            cursor_pixmap.fill(Qt.transparent)  # Otherwise you get a black background
            painter_cursor = QPainter(cursor_pixmap)

            brush_color = QColor(255, 255, 255, 50)

            painter_cursor.setBrush(brush_color)

            painter_cursor.drawEllipse(0, 0, brush_size_x, brush_size_y)
            painter_cursor.end()

            m_Cursor = QCursor(cursor_pixmap)
            self.setCursor(m_Cursor)

        else:
            self.setCursor(Qt.ArrowCursor)

class Canvas(QGraphicsView):

    config = {
        # Drawing options.
        'size': 1,
        'opacity': 0.5
    }

    def initialize(self, graphViewObject):

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.graphView = graphViewObject
        self.first_run = True

        self.array = np.array([])
        self.probability_image = np.array([])

    def loadImage(self):

        self.imlayer = QGraphicsPixmapItem(self.graphView.image)
        self.addlayer(self.imlayer)
        self.updateviewer()

        temp = QPixmap(self.graphView.image.size())
        temp.fill(Qt.transparent)

        self.result_layer = QGraphicsPixmapItem(temp)
        self.result_layer.setOpacity(self.config['opacity'])
        self.addlayer(self.result_layer)

        self.probability_layer = QGraphicsPixmapItem(temp)
        self.probability_layer.setOpacity(0.0)
        self.addlayer(self.probability_layer)

    def update_opacity(self):
        self.result_layer.setOpacity(self.config['opacity'])

    def set_config(self, key, value):
        self.config[key] = value

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.updateviewer()

    def scaleImage(self, factor):
        self.scale(factor,factor)

    def addlayer(self, layer):
        self.scene.addItem(layer)

    def updateviewer(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatioByExpanding)

    def show_results(self):

        self.probability_layer.setOpacity(0)

        qimg = qimage2ndarray.array2qimage(self.results)
        self.pixmap = QPixmap.fromImage(qimg)
        self.result_layer.setPixmap(self.pixmap)

    def show_probability_image(self,idx):

        self.probability_layer.setOpacity(1)

        class_probability_image = self.probability_image[:, :, idx]

        qimg = qimage2ndarray.array2qimage(class_probability_image, normalize=(0, 1))
        self.pixmap_prob = QPixmap.fromImage(qimg)

        self.probability_layer.setPixmap(self.pixmap_prob)


class DialogDemo(QDialog,Ui_Dialog):
    def __init__(self, *args, **kwargs):

        super(DialogDemo, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.resulting = False

        def accept():
            self.resulting = True

        def reject():
            self.resulting = False


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):

        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.graphView = GraphView()
        self.graphView.initialize()
        self.graphView.setMouseTracking(True)
        self.graphView.setFocusPolicy(Qt.StrongFocus)
        self.horizontalLayout.addWidget(self.graphView)

        self.graphView.autoRunBtn = self.autoRunButton
        self.graphView.run_results = self.run_results

        self.styleBarBlue = '''
                QProgressBar
                {
                    border: 0.5px solid grey;
                    border-radius: 5px;
                    text-align: center;
                    margin: 0.5px;
                    height: 1px;
                }
                QProgressBar::chunk
                {
                    background: #107BC6;
                    border-radius: 5px;
                    margin: 0.5px;
                    width: 5px;
                }
                '''
        self.styleBarGreen = '''
                QProgressBar
                {
                    border: 0.5px solid grey;
                    border-radius: 5px;
                    text-align: center;
                    margin: 0.5px;
                    height: 1px;
                }
                QProgressBar::chunk
                {
                    background: #4F9D4A;
                    border-radius: 5px;
                    margin: 0.5px;

                }
                '''

        self.progressBar.setValue(0)
        self.progressBar.setMinimum(0)
        self.progressBar.setStyleSheet(self.styleBarBlue)
        self.progressBar.setTextVisible(False)

        self.canvas = Canvas()
        self.canvas.initialize(self.graphView)
        self.horizontalLayout.addWidget(self.canvas)

        # Initialize dictionary segmenation model responsible for processing image
        self.dictSegModel = dsm.dictionarySegmentationModel()

        self.dialog = DialogDemo(self)
        self.dialog.initSpinBox.setProperty("value", self.dictSegModel.n_patches)
        self.dialog.clusterSpinBox.setProperty("value", self.dictSegModel.n_clusters)
        self.dialog.patchSpinBox.setProperty("value", self.dictSegModel.patch_size)

        self.dialog.buttonBox.accepted.connect(self.dialog.accept)
        self.dialog.buttonBox.rejected.connect(self.dialog.reject)

        # Setup the mode buttons
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)

        for mode in MODES:
            btn = getattr(self, '%sButton' % mode)
            btn.pressed.connect(lambda mode=mode: self.graphView.set_mode(mode))
            btn.pressed.connect(self.graphView.set_brush_cursor)
            mode_group.addButton(btn)

        self.runButton.pressed.connect(self.run_results)
        self.settingsButton.pressed.connect(self.show_options_window)

        # Setup the color selection buttons.
        self.primaryButton.pressed.connect(lambda: self.choose_color(self.set_primary_color))

        # Initialize button colours.
        for n, hex in enumerate(COLORS, 1):
            btn = getattr(self, 'colorButton_%d' % n)
            btn.setStyleSheet('QPushButton {\n    border: 1px solid #555;\n    border-radius: 5px;\n    border-style: solid;\n    padding: 5px;\n background-color: %s;\n }' % hex)
            btn.hex = hex  # For use in the event below

            def patch_mousePressEvent(self_, e):
                if e.button() == Qt.LeftButton:
                    self.set_primary_color(self_.hex)
                    self.graphView.set_brush_cursor()

            btn.mousePressEvent = types.MethodType(patch_mousePressEvent, btn)

        # Setup to agree with Canvas.
        self.set_primary_color('#C10000')

        # Set up select Qcombobox
        self.comboBox.addItem("None")
        self.comboBox.SizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.comboBox.activated.connect(self.show_probability_image)

        # Menu options
        self.actionNewImage.triggered.connect(self.reset_views)
        self.actionOpenImage.triggered.connect(self.open_file)
        self.actionSaveImage.triggered.connect(self.save_file)
        self.actionNewImage.triggered.connect(self.reset_views)
        self.actionLoadLabels.triggered.connect(self.open_labels)
        self.actionSaveLabels.triggered.connect(self.save_labels)

        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=True, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=True, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=True, triggered=self.normalSize)

        # Construct the sliders for brush size and opacity
        self.sizeselect = self.sizeSlider
        self.sizeselect.setRange(1, 50)
        self.sizeselect.setOrientation(Qt.Horizontal)
        self.sizeselect.valueChanged.connect(lambda s: self.graphView.set_config('size', s))
        self.sizeselect.valueChanged.connect(self.graphView.set_brush_cursor)
        self.sizeselect.setValue(BRUSH_INI_SIZE)

        self.opacity_select = self.opacitySlider
        self.opacity_select.setRange(0, 100)
        self.opacity_select.setValue(50)
        self.opacity_select.setOrientation(Qt.Horizontal)
        self.opacity_select.valueChanged.connect(lambda s: self.graphView.set_config('opacity', s/100))
        self.opacity_select.valueChanged.connect(lambda s: self.canvas.set_config('opacity', s/100))
        self.opacity_select.valueChanged.connect(self.update_opacity)

        # Add options to the menu
        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.menuBar.addMenu(self.viewMenu)

        self.show()

    def show_options_window(self):

            if self.dialog.exec_() == self.dialog.Accepted:

                if bool(self.dialog.patchSpinBox.value() % 2):

                    self.dictSegModel.n_patches = self.dialog.initSpinBox.value()
                    self.dictSegModel.n_clusters = self.dialog.clusterSpinBox.value()
                    self.dictSegModel.patch_size = self.dialog.patchSpinBox.value()

                    if self.graphView.image_is_uploaded:
                        self.progressBar.setStyleSheet(self.styleBarBlue)
                        self.progressBar.setMaximum(0)
                        self.preProcessThread.start()

                else:
                    QMessageBox.about(self, "Odd number", "Patch size needs to be an odd number. Your settings were not saved, try again.")
                    self.dialog.initSpinBox.setProperty("value", self.dictSegModel.n_patches)
                    self.dialog.clusterSpinBox.setProperty("value", self.dictSegModel.n_clusters)
                    self.dialog.patchSpinBox.setProperty("value", self.dictSegModel.patch_size)
            else:
                self.dialog.initSpinBox.setProperty("value", self.dictSegModel.n_patches)
                self.dialog.clusterSpinBox.setProperty("value", self.dictSegModel.n_clusters)
                self.dialog.patchSpinBox.setProperty("value", self.dictSegModel.patch_size)

    def run_results(self):
        if self.graphView.image_is_uploaded:
            if not self.preProcessThread.isRunning():

                # Get options from user for the dictionary segmenation
                self.dictSegModel.two_step_diffusion = self.diffusionCheckBox.isChecked()
                self.dictSegModel.overwrite = self.overwriteCheckBox.isChecked()
                self.dictSegModel.binarisation = self.binarisationCheckBox.isChecked()

                # Extract label image and convert to Qimage
                label_image = self.graphView.roilayer.pixmap().toImage()
                label_image = label_image.convertToFormat(QImage.Format_RGB32)
                self.array = qimage2ndarray.rgb_view(label_image)

                # Prepare the labels and run iteration
                self.dictSegModel.prepare_labels(self.array)
                self.dictSegModel.iterate_dictionary()

                # Extract the segmentation image
                self.canvas.results = self.dictSegModel.segmentation_image

                # Extract probability image
                self.canvas.probability_image = self.dictSegModel.probability_image

                self.canvas.show_results()

                self.update_class_selection()
            else:
                QMessageBox.about(self, "Patience is key", "Still Preprocessing the Image, Please take a chill pill")
        else:
            QMessageBox.about(self, "No Image", "Need to upload a image")

    def update_class_selection(self):
        color_reference = self.dictSegModel.color_reference

        self.comboBox.clear()
        self.comboBox.addItem('None')

        pixmap_icon = QPixmap(20, 20)

        for key in color_reference.keys():

            text = 'Class: {}'.format(key + 1)

            color_rgb = color_reference[key]
            pixmap_icon.fill(QColor(color_rgb[0],color_rgb[1],color_rgb[2]))

            self.comboBox.addItem(QIcon(pixmap_icon),text)

    def show_probability_image(self):

        if (self.comboBox.currentText()) == 'None':
            self.canvas.show_results()
        else:
            idx = self.comboBox.currentIndex() - 1
            self.canvas.show_probability_image(idx)

    def choose_color(self, callback):
        dlg = QColorDialog()
        if dlg.exec():
            callback(dlg.selectedColor().name())

    def set_primary_color(self, hex):
        self.graphView.set_primary_color(hex)
        self.primaryButton.setStyleSheet('QPushButton { background-color: %s; }' % hex)

    def zoomIn(self):
        self.graphView.zoomIn()
        self.canvas.zoomIn()

    def zoomOut(self):
        self.graphView.zoomOut()
        self.canvas.zoomOut()

    def normalSize(self):
        self.graphView.normalSize()
        self.canvas.normalSize()

    def reset_views(self):
        self.graphView.initialize()
        self.canvas.initialize(self.graphView)

    def update_opacity(self):

        if self.graphView.image_is_uploaded:
            self.graphView.update_opacity()
            self.canvas.update_opacity()

    def save_file(self):
        """
        Save active canvas to image file.
        :return:
        """
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "PNG Image file (*.png)")

        if path:
            pixmap_out = self.canvas.pixmap
            pixmap_out.save(path, "PNG" )

    def save_labels(self):
        """
        Save active canvas to image file.
        :return:
        """
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "PNG Image file (*.png)")

        if path:
            pixmap_out = self.graphView.roilayer.pixmap()
            pixmap_out.save(path, "PNG" )

    def open_file(self):
        """
        Open target image
        """
        self.reset_views()

        path, _ = QFileDialog.getOpenFileName(self, "Open file", "", "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")

        if path:

            pixmap = QPixmap()
            pixmap.load(path)

            self.dictSegModel.load_image(path)
            self.progressBar.setMaximum(0)
            self.preProcessThread = PreprocessImageThread(self.dictSegModel)
            self.preProcessThread.finished.connect(self.finished_preprocess)
            self.preProcessThread.start()

            self.graphView.loadImage(pixmap)
            self.canvas.loadImage()

            self.graphView.image_is_uploaded = True
            self.graphView.set_brush_cursor()

    def open_labels(self):
        """
        Open label image
        """
        if self.graphView.image_is_uploaded:

            path, _ = QFileDialog.getOpenFileName(self, "Open file", "", "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")

            if path:
                pixmap = QPixmap()
                pixmap.load(path)

                if (pixmap.size() == self.graphView.image.size()):

                    self.graphView.loadLabels(pixmap)
                else:
                    QMessageBox.about(self, "Size does not match", "Please load an label image that is the same size as the target image")
        else:
            QMessageBox.about(self, "Not yet", "Please open an image to segment first")

    def finished_preprocess(self):
        self.progressBar.setStyleSheet(self.styleBarGreen)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(100)


class PreprocessImageThread(QThread):

    def __init__(self, dictSegModel):
        QThread.__init__(self)
        self.object = dictSegModel

    def __del__(self):
        self.wait()

    def run(self):
        self.object.preprocess()


if __name__ == '__main__':

    app = QApplication([])
    window = MainWindow()
    app.exec_()
