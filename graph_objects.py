import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Union, Tuple, Dict, Any, Callable, List
import numpy as np
import numpy.typing as npt
from transitions import lab_to_graph, graph_to_lab
from math import *

GRAPH_PARAMS = {
    "SC_WIDTH": 1200,
    "SC_HEIGHT": 800,
    "SIDE_WIDTH": 600,
    "METER": 500,
    "TABLE_SURF_COLS": ['id', 'Length', 'Slope', "Start Point"],
    "TABLE_BODY_COLS": ['id', 'Mass', 'Geometry', 'Angle', "Position", 'Velocity', 'k energy'],
}


class Scene(QtWidgets.QGraphicsScene):

    def __init__(self, w, h):
        super(Scene, self).__init__(0, 0, w, h)


# class LabParams(QtWidgets.QWidget):
#
#     def __init__(self, params):
#         super(LabParams, self).__init__()
#         self.form = QtWidgets.QVBoxLayout()
#         self.first_row =  QtWidgets.QHBoxLayout()
#         # self.second_row = QtWidgets.QHBoxLayout()
#         self.title = QtWidgets.QLabel()
#         self.title.setText('Environment parameters')
#
#         speed_form = QtWidgets.QFormLayout()
#         self.speed_input = QtWidgets.QSpinBox()
#         speed_form.addRow('Speed', self.speed_input)
#         speed_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
#         self.speed_input.setRange(params[0]['min'], params[0]['max'])
#         self.speed_input.setSingleStep(params[0]['step'])
#         self.speed_input.setSuffix(' tacts per sec')
#         self.speed_input.setValue(params[0]['init'])
#
#         gravity_form = QtWidgets.QFormLayout()
#         self.gravity_input = QtWidgets.QCheckBox()
#         self.gravity_input.setChecked(True)
#         gravity_form.addRow('Add gravity', self.gravity_input)
#         gravity_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
#
#         substance_form = QtWidgets.QFormLayout()
#         self.substance_input = QtWidgets.QComboBox()
#         for sub in params[1]:
#             self.substance_input.addItem(sub)
#         substance_form.addRow('Fill substance: ', self.substance_input)
#         substance_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
#
#         self.first_row.addLayout(speed_form)
#         self.first_row.addLayout(gravity_form)
#         self.first_row.addLayout(substance_form)
#         self.form.addWidget(self.title)
#         self.form.addLayout(self.first_row)
#         # self.form.addLayout(self.second_row)
#         self.setLayout(self.form)


class CreateNumericItemForm(QtWidgets.QWidget):

    def __init__(self, item_name: str, item_dict: Dict):
        super(CreateNumericItemForm, self).__init__()
        form = QtWidgets.QFormLayout()
        self.type = item_dict['type']
        if item_dict['type'] == 'float':
            self.input = QtWidgets.QDoubleSpinBox()
            self.input.setDecimals(2)
        elif item_dict['type'] == 'int':
            self.input = QtWidgets.QSpinBox()
        form.addRow(item_name, self.input)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.input.setRange(item_dict['min'], item_dict['max'])
        self.input.setSingleStep(item_dict['step'])
        self.input.setSuffix(item_dict['measure'])
        self.input.setValue(item_dict['init'])
        self.setLayout(form)


class CreateComboItemForm(QtWidgets.QWidget):

    def __init__(self, item_name: str, item_dict: Dict):
        super(CreateComboItemForm, self).__init__()
        form = QtWidgets.QFormLayout()
        self.input = QtWidgets.QComboBox()
        for type in item_dict['data']:
            self.input.addItem(type)
        form.addRow(item_name, self.input)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.setLayout(form)


class CreateCheckItemForm(QtWidgets.QWidget):

    def __init__(self, item_name: str, item_dict: Dict):
        super(CreateCheckItemForm, self).__init__()
        form = QtWidgets.QFormLayout()
        self.input = QtWidgets.QCheckBox()
        self.input.setChecked(item_dict['init'])
        form.addRow(item_name, self.input)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
        self.setLayout(form)


class CreateInputForm(QtWidgets.QWidget):

    def __init__(self, name: str, items: Dict, slot: Callable):
        super(CreateInputForm, self).__init__()
        self.setObjectName(name)
        self.layout = QtWidgets.QVBoxLayout()
        self.form_1 = QtWidgets.QHBoxLayout()
        self.form_2 = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(name)
        self.form_1.addWidget(label)
        i = 0
        for item_name, item_data in items.items():
            if not item_data:
                continue
            if item_data['type'] in ('int', 'float'):
                input = CreateNumericItemForm(item_name, item_data)
            elif item_data['type'] == 'combo':
                input = CreateComboItemForm(item_name, item_data)
            elif item_data['type'] == 'check':
                input = CreateCheckItemForm(item_name, item_data)
            else:
                input = QtWidgets.QLineEdit()
            input.setObjectName(item_name)
            i += 1
            if i < 6:
                self.form_1.addWidget(input)
            else:
                self.form_2.addWidget(input)
        self.btn = QtWidgets.QPushButton()
        self.btn.setText(f'Create {name}')
        self.btn.clicked.connect(slot)
        self.form_1.addWidget(self.btn)
        self.layout.addLayout(self.form_1)
        self.layout.addLayout(self.form_2)
        self.setLayout(self.layout)


class SidePanel(QtWidgets.QWidget):

    def __init__(self, w: int, lab_params: Dict, lab_slot: Callable, surf_params: Dict, surf_slot: Callable,
                 body_params: Dict, body_slot: Callable, box_params: Dict, box_slot: Callable):
        super(SidePanel, self).__init__()
        self.setFixedWidth(w)
        self.layout = QtWidgets.QVBoxLayout(self)

        self.create_surface = CreateInputForm('Surface', surf_params, surf_slot)
        self.table_of_surfaces = QtWidgets.QTableView()
        self.btn_del_surf = QtWidgets.QPushButton('Delete surface')

        self.create_box = CreateInputForm('Box', box_params, box_slot)
        # self.btn_del_box = QtWidgets.QPushButton('Delete box')

        self.create_body = CreateInputForm('Body', body_params, body_slot)
        self.table_of_bodies = QtWidgets.QTableView()
        self.btn_del_body = QtWidgets.QPushButton('Delete body')

        self.table_of_contacts = QtWidgets.QTableView()

        self.create_lab = CreateInputForm('Lab', lab_params, lab_slot)
        self.btn_start = QtWidgets.QPushButton('Start')

        self.layout.addWidget(self.create_surface)
        self.layout.addWidget(self.table_of_surfaces)
        self.layout.addWidget(self.btn_del_surf)
        self.layout.addWidget(self.create_box)
        # self.layout.addWidget(self.btn_del_box)
        self.layout.addWidget(self.create_body)
        self.layout.addWidget(self.table_of_bodies)
        self.layout.addWidget(self.btn_del_body)
        self.layout.addWidget(self.table_of_contacts)
        self.layout.addSpacing(20)
        self.layout.addWidget(self.create_lab)
        self.layout.addWidget(self.btn_start)

    def get_input_data(self, item_name):
        input_form = self.findChild(QtWidgets.QWidget, item_name)
        data_list = []
        for input_item in input_form.children():
            if input_item.__class__ == CreateNumericItemForm:
                if input_item.type == 'float':
                    val = round(input_item.input.value(), 3)
                elif input_item.type == 'int':
                    val = input_item.input.value()
                data_list.append(val)
            elif input_item.__class__ == CreateComboItemForm:
                data_list.append(input_item.input.currentText())
            elif input_item.__class__ == CreateCheckItemForm:
                data_list.append(input_item.input.isChecked())
        return data_list


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, h_headers, data=None):
        super().__init__()
        self.horizontal_headers = ['']*len(h_headers)
        for i, h_h in enumerate(h_headers):
            self.setHeaderData(i, QtCore.Qt.Orientation.Horizontal, h_h)
        if not data:
            self._data = [[''] * len(h_headers)]
        else:
            self._data = data

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QtCore.QModelIndex()):
        # return len(max(self._data, key=len))
        return len(self.horizontal_headers)

    def setHeaderData(self, section: int, orientation: QtCore.Qt.Orientation, value: Any,
                      role: int = QtCore.Qt.ItemDataRole.EditRole) -> bool:
        if orientation == QtCore.Qt.Orientation.Horizontal \
                and role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
            try:
                self.horizontal_headers[section] = value
                return True
            except:
                return False
        return super().setHeaderData(section, orientation, value, role)

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation,
                   role: int = QtCore.Qt.ItemDataRole.EditRole) -> Any:
        # if role != QtCore.Qt.DisplayRole:
        #     return QtCore.QVariant()
        if orientation == QtCore.Qt.Orientation.Horizontal \
                and role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            try:
                return self.horizontal_headers[section]
            except:
                pass
        # return QtCore.QString(self.header[section-1])
        return super().headerData(section, orientation, role)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            try:
                return self._data[index.row()][index.column()]
            except IndexError:
                return ''

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            if not value:
                return False
            self._data[index.row()][index.column()] = value
            self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        return super().flags(index) | QtCore.Qt.ItemIsEditable

    def insertRows(self, position, rows, parent=QtCore.QModelIndex()):
        position = (position + self.rowCount()) if position < 0 else position
        start = position
        end = position + rows - 1
        self.beginInsertRows(parent, start, end)
        self._data.append(['']*self.columnCount())
        self.endInsertRows()
        return True

    def removeRows(self, position, rows, parent=QtCore.QModelIndex()):
        start, end = position, rows
        self.beginRemoveRows(parent, start, end)
        del self._data[start:end + 1]
        self.endRemoveRows()
        return True


class TableItemModel(QtGui.QStandardItemModel):
    def __init__(self, headers):
        super(TableItemModel, self).__init__()
        self.setHorizontalHeaderLabels(headers)


class GraphBall(QtWidgets.QGraphicsObject):
    replaced = QtCore.pyqtSignal(QtWidgets.QGraphicsObject)

    def __init__(self, r):
        super(GraphBall, self).__init__()
        self.r = r * GRAPH_PARAMS["METER"]
        self.rect = QtCore.QRectF(0, 0, 2 * self.r, 2 * self.r)
        self.ellipse = QtWidgets.QGraphicsEllipseItem(self.rect)
        self.name = "A ball"
        self.mark = QtWidgets.QGraphicsLineItem(self)
        self.mark.setLine(self.r, self.r, 2 * self.r, self.r)
        self.legend = QtWidgets.QGraphicsSimpleTextItem(self)
        self.legend.setText(self.name)
        self.legend.setPos(0, -25)
        self.start_move = None
        self.busy = False

    def paint(self, painter, option, widget):
        # painter.drawEllipse(self.rect)
        return self.ellipse.paint(painter, option, widget)

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect

    def set_name(self, name):
        self.name = name
        self.legend.setText(name)

    def __repr__(self):
        return self.name

    @property
    def center(self):
        return self.pos() + QtCore.QPointF(self.r, self.r)

    @property
    def center_lab(self):
        return graph_to_lab(self.center)

    def set_lab_pos(self, lab_pos: Union[npt.NDArray, Tuple[float, float]]):
        self.setPos(lab_to_graph(lab_pos) - QtCore.QPointF(self.r, self.r))

    def set_busy(self, option: bool):
        self.busy = option

    def mousePressEvent(self, event):
        if event.button() == 1:
            self.setSelected(True)
            if not self.busy:
                self.start_move = event.pos()
        elif event.button() == 0:
            self.setSelected(False)

    def mouseMoveEvent(self, event):
        if self.start_move:
            self.legend.setText(self.center_lab[2])
            self.setPos(self.pos() + (event.pos() - self.start_move))

    def mouseReleaseEvent(self, event):
        if not self.busy:
            self.legend.setText(self.name)
            self.replaced.emit(self)
            self.start_move = None


class GraphicsBody(QtWidgets.QGraphicsObject):
    replaced = QtCore.pyqtSignal(QtWidgets.QGraphicsObject)
    is_selected = QtCore.pyqtSignal(QtWidgets.QGraphicsObject)
    is_moving = QtCore.pyqtSignal(QtWidgets.QGraphicsObject)

    def __init__(self, movable_type: str, geometry: Tuple[float, float], position: QtCore.QPointF, angle: float):
        super(GraphicsBody, self).__init__()
        self.type = movable_type
        self.angle = angle
        if self.type == 'BRICK':
            self.rect = QtCore.QRectF(0, 0, *geometry)
            self.my_shape = QtWidgets.QGraphicsRectItem(self.rect)
            self.anchor_vector = QtCore.QPointF(0.0, geometry[1])
        elif self.type in ('BALL', 'RING', 'WHEEL', 'NO_SPIN'):
            self.r = geometry[0]
            self.rect = QtCore.QRectF(0, 0, 2 * self.r, 2 * self.r)
            self.anchor_vector = QtCore.QPointF(self.r, self.r)
            self.my_shape = QtWidgets.QGraphicsEllipseItem(self.rect)
            if self.type != 'NO_SPIN':
                self.mark = QtWidgets.QGraphicsLineItem(self)
                self.mark.setLine(self.r, self.r, 2 * self.r, self.r)
        self.setTransformOriginPoint(self.anchor_vector)
        self.set_pos(position)
        # self.setPos(position - self.anchor_vector)
        self.setRotation(-self.angle*180)
        self.name = f"A {self.type}"
        self.legend = QtWidgets.QGraphicsSimpleTextItem(self)
        self.legend.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self.legend.setText(self.name)
        self.legend.setPos(0, -25)
        self.start_move = None
        self.busy = False

    def paint(self, painter, option, widget):
        return self.my_shape.paint(painter, option, widget)

    def boundingRect(self) -> QtCore.QRectF:
        return self.my_shape.boundingRect()

    def shape(self) -> QtGui.QPainterPath:
        return self.my_shape.shape()

    def set_name(self, name):
        self.name = name
        self.legend.setText(name)

    def set_pos(self, pos):
        self.setPos(pos - self.anchor_vector)

    def set_angle(self, a):
        self.angle = a

    def set_legend(self, txt):
        self.legend.setText(txt)

    def __repr__(self):
        return self.name

    @property
    def anchor_pos(self):
        return self.pos() + self.anchor_vector

    def set_busy(self, option: bool):
        self.busy = option

    def mousePressEvent(self, event):
        if event.button() == 1:
            self.setSelected(True)
            self.is_selected.emit(self)
            if not self.busy:
                self.start_move = event.pos()
        elif event.button() == 0:
            pass

    def mouseMoveEvent(self, event):
        if self.start_move and not self.busy:
            self.setPos(self.pos() + (event.pos() - self.start_move))
            self.is_moving.emit(self)

    def mouseReleaseEvent(self, event):
        if not self.busy:
            self.legend.setText(self.name)
            self.replaced.emit(self)
            self.start_move = None


class GraphicsSurface(QtWidgets.QGraphicsObject):

    def __init__(self, start: QtCore.QPointF, finish: QtCore.QPointF, length: float, slope: float):
        super(GraphicsSurface, self).__init__()
        self.start = start
        self.finish = finish
        self.slope = slope
        self.length = length
        self.name = 'Surface'
        start_x, start_y = start.x(), start.y()
        finish_x, finish_y = finish.x(), finish.y()
        self.tau = (finish - start)/length
        self.norm = QtCore.QPointF(self.tau.y(), - self.tau.x())

        self.my_shape = QtWidgets.QGraphicsLineItem(start_x, start_y, finish_x, finish_y)

    def paint(self, painter, option, widget):
        return self.my_shape.paint(painter, option, widget)

    def boundingRect(self) -> QtCore.QRectF:
        return self.my_shape.boundingRect()

    def shape(self):
        return self.my_shape.shape()

    def set_name(self, name):
        self.name = name

    def _dist_to_line(self, point: QtCore.QPointF) -> float:
        one = point - self.start
        two = self.finish - point
        return (one.y() * two.x() - two.y() * one.x())/self.length  # np.sign(one.y()) * np.sign(two.y())

    # def in_section(self, point: QtCore.QPointF) -> bool:
    #     dist, flag = self._dist_to_line(point)
    #     return abs(dist) <= 1 and flag >= 0

    def distance(self, point: QtCore.QPointF) -> Tuple:
        epi_point = point + self.norm * self._dist_to_line(point)
        if self.contains(epi_point):
            return self._dist_to_line(point), epi_point
        to_start = sqrt(QtCore.QPointF.dotProduct(point - self.start, point - self.start))
        to_finish = sqrt(QtCore.QPointF.dotProduct(point - self.finish, point - self.finish))
        return (to_start, self.start) if to_start <= to_finish else (to_finish, self.finish)

    def body_contact_surface(self, body: GraphicsBody) -> Any:
        if body.type == 'BRICK':
            if self.slope != body.angle:
                return False
            return sqrt(QtCore.QPointF.dotProduct(body.anchor_pos - self.start, body.anchor_pos - self.start)) \
                if self.my_shape.contains(body.anchor_pos) else False
        elif body.type in ('BALL', 'RING', 'WHEEL'):
            dist, print = self.distance(body.anchor_pos)
            return sqrt(QtCore.QPointF.dotProduct(print - self.start, print - self.start)) \
                if abs(abs(dist) - body.r) <= 1 else False
        else:
            return False


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.scene = Scene()
        self.view = QtWidgets.QGraphicsView(self.scene)
        container = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QHBoxLayout()
        side_widget = QtWidgets.QWidget()
        side_widget.setFixedWidth(GRAPH_PARAMS["SIDE_WIDTH"])
        side_layout = QtWidgets.QVBoxLayout()

        self.table = QtWidgets.QTableView()
        data_model = TableItemModel()
        self.table.setModel(data_model)
        side_layout.addWidget(self.table)

        side_widget.setLayout(side_layout)
        self.layout.addWidget(side_widget)
        self.layout.addWidget(self.view)
        container.setLayout(self.layout)

        desktop = QtWidgets.QDesktopWidget().screenGeometry()
        self.setGeometry(
            (desktop.width() - (GRAPH_PARAMS["SC_WIDTH"] + GRAPH_PARAMS["SIDE_WIDTH"] + 30)) // 2,
            (desktop.height() - (GRAPH_PARAMS["SC_HEIGHT"] + 30)) // 2,
            GRAPH_PARAMS["SC_WIDTH"] + GRAPH_PARAMS["SIDE_WIDTH"] + 30,
            GRAPH_PARAMS["SC_HEIGHT"] + 30
        )

        ball = GraphBall(0.07)
        ball.setPos(lab_to_graph((0.5, 0.5)) - QtCore.QPointF(ball.r, ball.r))
        self.scene.addItem(ball)
        ball.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemIsSelectable)

        rect = QtWidgets.QGraphicsRectItem(0, 0, 200, 50)
        rect.setPos(500, 200)
        self.scene.addItem(rect)
        rect.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemIsSelectable)

        self.setCentralWidget(container)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

