import sys
from PyQt5 import Qt, QtCore, QtGui, QtWidgets
from collections import defaultdict
from typing import List
from forces import PreSubs, ContactBrickS
from graph_objects import Scene, SidePanel, TableItemModel, GraphicsBody, GraphicsSurface
from transitions import (SC_WIDTH, SC_HEIGHT, SIDE_WIDTH, input_body_to_graph,
                         TABLE_SURF_INPUTS, TABLE_SURF_HEADERS, TABLE_BOX_INPUTS, TABLE_BOX_HEADERS, TABLE_BODY_INPUTS,
                         from_pixels, lab_to_tab, TABLE_BODY_HEADERS, TABLE_CONTACT_HEADERS, body_legend_pos,
                         input_surf_to_graph, lab_to_graph)
from objects import Ball, Brick, Surface, MovableTypes
# from Lab import LabSystem, LAB_PARAMS
from Lab import LabSystem, LAB_HEADERS, LAB_INPUTS


class Worker(QtCore.QThread):
    step_done = QtCore.pyqtSignal(dict)
    all_done = QtCore.pyqtSignal(bool)

    def __init__(self, lab, steps: int):
        super(Worker, self).__init__()
        self.lab = lab
        self.steps = steps

    def run(self):
        while ...:
            ans = {}
            # ans = self.lab.step_ball()
            ans = self.lab.step_body()
            self.step_done.emit(ans)
            self.msleep(10)
        self.all_done.emit(True)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.model_of_surfaces = TableItemModel(TABLE_SURF_HEADERS)
        self.model_of_bodies = TableItemModel(TABLE_BODY_HEADERS)
        self.model_of_contacts = TableItemModel(TABLE_CONTACT_HEADERS)
        self.surfaces = []
        self.bodies = []
        self.contacts = []
        self.surf_count = 0
        self.body_count = defaultdict(int)

        self.scene = Scene(SC_WIDTH, SC_HEIGHT)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)

        self.side = SidePanel(SIDE_WIDTH, LAB_INPUTS, self.lab_creation, TABLE_SURF_INPUTS, self.surface_creation,
                              TABLE_BODY_INPUTS, self.body_creation, TABLE_BOX_INPUTS, self.box_creation)
        self.side.table_of_surfaces.setModel(self.model_of_surfaces)
        self.side.table_of_surfaces.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked)
        self.side.table_of_surfaces.clicked.connect(self.surface_selection)
        self.side.btn_del_surf.clicked.connect(self.delete_surf)

        self.side.table_of_bodies.setModel(self.model_of_bodies)
        self.side.table_of_bodies.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked)
        self.side.table_of_bodies.clicked.connect(self.body_tab_selection)
        self.side.table_of_bodies.doubleClicked.connect(self.body_table_edit)
        column_sizes = [60, 60, 30, 30, 30, 40, 50, 50, 60, 60, 60, 60]
        for col in range(11):
            self.side.table_of_bodies.setColumnWidth(col, column_sizes[col])
        self.side.btn_del_body.clicked.connect(self.delete_body)

        self.side.table_of_contacts.setModel(self.model_of_contacts)
        self.side.table_of_contacts.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked)
        self.side.table_of_contacts.clicked.connect(self.contact_tab_selection)
        self.side.table_of_contacts.doubleClicked.connect(self.contact_table_edit)

        self.side.btn_start.clicked.connect(self.start)

        # self.side.btn_create_lab.clicked.connect(self.lab_creation)

        central_widget = QtWidgets.QWidget()
        desktop = QtWidgets.QDesktopWidget().screenGeometry()
        self.setGeometry(
            (desktop.width() - (SC_WIDTH + SIDE_WIDTH + 30)) // 2,
            (desktop.height() - (SC_HEIGHT + 21)) // 2,
            SC_WIDTH + SIDE_WIDTH + 30,
            SC_HEIGHT + 21
        )
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.side)
        layout.addWidget(self.view)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.lab = None
        self.thread = None

    def surface_creation(self, data: List = None):
        if data:
            data_list = data
        else:
            data_list = self.side.get_input_data('Surface')
        self.surf_count += 1
        self.model_of_surfaces.appendRow([QtGui.QStandardItem(str(item)) for item in data_list])
        new_surface = GraphicsSurface(*input_surf_to_graph(data_list))
        new_surface.set_name(f'Surface-{self.surf_count}')
        self.scene.addItem(new_surface)
        # new_surface.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.surfaces.append(new_surface)

    def surface_selection(self, index):
        for it in self.scene.selectedItems():
            it.setSelected(False)
        self.surfaces[index.row()].setSelected(True)

    def delete_surf(self):
        if selected := self.side.table_of_surfaces.selectedIndexes():
            self.model_of_surfaces.removeRow(selected[0].row())
            self.scene.removeItem(self.surfaces.pop(selected[0].row()))

    def box_creation(self):
        data_list = self.side.get_input_data('Box')
        w, h, x, y = data_list
        self.surface_creation([w, 0.0, x, y])
        self.surface_creation([h, 0.5, x + w, y])
        self.surface_creation([w, 1.0, x + w, y + h])
        self.surface_creation([h, 1.5, x, y + h])

    def body_creation(self):
        data_list = self.side.get_input_data('Body')
        self.body_count[data_list[0]] += 1
        self.model_of_bodies.appendRow([QtGui.QStandardItem(str(item)) for item in data_list])
        new_body = GraphicsBody(*input_body_to_graph(data_list))
        new_body.set_name(f"{new_body.type}-{self.body_count[new_body.type]}")
        new_body.is_selected.connect(self.body_scene_selected)
        new_body.is_moving.connect(self.body_scene_moving)
        new_body.replaced.connect(self.body_model_edit)
        self.scene.addItem(new_body)
        new_body.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.bodies.append(new_body)

    def contact_creation(self, *args):
        self.model_of_contacts.appendRow([QtGui.QStandardItem(item) for item in args])
        # self.contacts.append()

    def body_scene_selected(self, body):
        for row in range(self.model_of_contacts.rowCount()):
            if self.model_of_contacts.item(row, TABLE_CONTACT_HEADERS.index('Body')).text() == body.name:
                self.model_of_contacts.removeRow(row)
        self.side.table_of_bodies.selectRow(self.bodies.index(body))

    def body_tab_selection(self, index):
        for it in self.scene.selectedItems():
            it.setSelected(False)
        self.bodies[index.row()].setSelected(True)

    def contact_tab_selection(self, index):
        for it in self.scene.selectedItems():
            it.setSelected(False)
        body_name = self.model_of_contacts.item(index.row(), TABLE_CONTACT_HEADERS.index('Body')).text()
        surface_name = self.model_of_contacts.item(index.row(), TABLE_CONTACT_HEADERS.index('Surface')).text()
        body_item = [body for body in self.bodies if body.name == body_name][0]
        surface_item = [surf for surf in self.surfaces if surf.name == surface_name][0]
        body_item.setSelected(True)
        surface_item.setSelected(True)

    def contact_table_edit(self, index):
        pass

    def ispc(self, e):
        print(e)

    def body_scene_moving(self, body):
        body.set_legend(body_legend_pos(body))

    def body_model_edit(self, body: GraphicsBody, new_v=None, new_ang_v=None):
        if not body.busy:
            for surf in self.surfaces:
                if cont := surf.body_contact_surface(body):
                    self.contact_creation(surf.name, body.name, str(from_pixels(cont)), '0')
        # print(body, new_v)
        row = self.bodies.index(body)
        col_x, col_y = TABLE_BODY_HEADERS.index('X'), TABLE_BODY_HEADERS.index('Y')
        col_ang = TABLE_BODY_HEADERS.index('Angle')
        new_x, new_y = body_legend_pos(body).split(',')
        new_ang = str(body.angle)
        self.model_of_bodies.setItem(row, col_x, QtGui.QStandardItem(new_x))
        self.model_of_bodies.setItem(row, col_y, QtGui.QStandardItem(new_y))
        self.model_of_bodies.setItem(row, col_ang, QtGui.QStandardItem(new_ang))
        # print('new pos sat')
        if new_v is not None:
            # print('new v', new_v)
            col_vx, col_vy = TABLE_BODY_HEADERS.index('Vx'), TABLE_BODY_HEADERS.index('Vy')
            new_vx, new_vy = lab_to_tab(new_v).split(',')
            # print(new_vx, new_vy )
            self.model_of_bodies.setItem(row, col_vx, QtGui.QStandardItem(new_vx))
            self.model_of_bodies.setItem(row, col_vy, QtGui.QStandardItem(new_vy))
        if new_ang_v is not None:
            col_ang_v = TABLE_BODY_HEADERS.index('Ang_v')
            new_ang_v = str(new_ang_v)
            self.model_of_bodies.setItem(row, col_ang_v, QtGui.QStandardItem(new_ang_v))

    def body_table_edit(self, e):
        row, col = e.row(), e.column()
        if col == TABLE_BODY_HEADERS.index('X'):
            new_x = float(self.model_of_bodies.item(row, col).text())
            new_y = float(self.model_of_bodies.item(row, col+1).text())
            self.bodies[row].set_pos(lab_to_graph((new_x, new_y)))
        elif col == TABLE_BODY_HEADERS.index('Y'):
            new_y = float(self.model_of_bodies.item(row, col).text())
            new_x = float(self.model_of_bodies.item(row, col-1).text())
            self.bodies[row].set_pos(lab_to_graph((new_x, new_y)))
        elif col == TABLE_BODY_HEADERS.index('Angle'):
            new_angle = float(self.model_of_bodies.item(row, col).text())
            # print(new_angle)
            self.bodies[row].setRotation(-new_angle*180.0)
            self.bodies[row].set_angle(new_angle)

    def delete_body(self):
        if selected := self.side.table_of_bodies.selectedIndexes():
            self.model_of_bodies.removeRow(selected[0].row())
            self.scene.removeItem(self.bodies.pop(selected[0].row()))

    def lab_body_creation(self, row):
        # print('body creation', row)
        data_list = [self.model_of_bodies.item(row, col).text() for col in range(len(TABLE_BODY_HEADERS))]
        body_type, body_mat, params = data_list[0], data_list[1], dict(zip(TABLE_BODY_HEADERS[2:], map(float, data_list[2:])))
        # print(body_type, body_mat, params)
        if body_type in ('BALL', 'RING', 'WHEEL', 'NO_SPIN'):
            new_body = Ball(params['Mass'], body_mat, (params['Geometry_1'], 0), body_type)
        elif body_type == 'BRICK':
            new_body = Brick(params['Mass'], body_mat, (params['Geometry_1'], params['Geometry_2']))
        else:
            return None

        new_body.set_name(self.bodies[row].name)
        new_body.set_angle(params['Angle'])
        new_body.set_pos((params['X'], params['Y']))
        new_body.set_v((params['Vx'], params['Vy']))
        # print('new_body', new_body.__dict__)
        # print('new body substant forces', new_body.substant_forces)
        # print(new_body)
        # print(new_body.__dict__)
        return new_body

    def lab_surf_creation(self, row):
        params = {value: float(self.model_of_surfaces.item(row, col).text())
                  for col, value in enumerate(TABLE_SURF_INPUTS)}
        new_surf = Surface(params['Length'])
        new_surf.set_name(self.surfaces[row].name)
        new_surf.set_pos(params['X0'], params['Y0'])
        new_surf.set_slope(params['Slope'])
        return new_surf

    def lab_contact_creation(self, row):
        params = {value: self.model_of_contacts.item(row, col).text()
                  for col, value in enumerate(TABLE_CONTACT_HEADERS)}
        # print('contact params', params)
        for key, val in params.items():
            # print(key, val)
            try:
                # print('val', val)
                float_val = float(val)
                # print('float val', float_val)
                params[key] = float_val
            except ValueError:
                continue
        # print('contact creation', params)
        return params

    def lab_creation(self):
        data_list = self.side.get_input_data('Lab')
        # print('lab data', data_list)
        lab = LabSystem(*data_list)
        lab.bodies = []
        # print('lab created', lab.bodies)
        surf_row_count = 0
        body_row_count = 0
        contact_row_count = 0
        if self.lab:
            body_row_count = len(self.lab.bodies)
            surf_row_count = len(self.lab.surfaces)
            contact_row_count = len(self.lab.contacts)
            # self.lab.bodies = []

        for row in range(surf_row_count, self.model_of_surfaces.rowCount()):
            # print('surf row', row)
            lab.add_surface(self.lab_surf_creation(row))
        for row in range(body_row_count, self.model_of_bodies.rowCount()):
            # print('body row', row)
            added_body = self.lab_body_creation(row)
            added_body.substant_forces = []
            added_body.contact_surfaces = []
            lab.add_body(added_body)
            for b in lab.bodies:
                print(b.name, b.substant_forces)
            self.bodies[row].set_busy(True)
        for row in range(contact_row_count, self.model_of_contacts.rowCount()):
            lab.add_contact(self.lab_contact_creation(row))
        # print(lab.__dict__)
        # print(lab.contacts)
        self.lab = lab

    def start(self):
        if self.thread:
            self.done(True)
        else:
            for body in self.scene.selectedItems():
                body.setSelected(False)
            self.thread = Worker(self.lab, 500)
            self.thread.step_done.connect(self.results)
            self.thread.all_done.connect(self.done)
            self.thread.start()
            self.side.btn_start.setText("Stop")

    def results(self, data):
        for body_name, state in data.items():
            # print(body_name, state)
            moved_body = [body for body in self.bodies if body.name == body_name].pop()
            # print(moved_body)
            moved_body.set_pos(lab_to_graph(state['position']))
            moved_body.set_angle(state['angle'])
            moved_body.setRotation(-state['angle']*180)
            # print('body moved')
            self.body_model_edit(moved_body, state['v'], state['ang_v'])
            # print('results')

    def done(self, e):
        self.thread.terminate()
        self.thread = None
        self.side.btn_start.setText("Start")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
