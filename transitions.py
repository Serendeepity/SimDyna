from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Union, Tuple
import numpy as np
import numpy.typing as npt
from math import *
from forces import SUBS
from objects import Movable, MovableTypes, Materials


METER = 100
SC_WIDTH = 1200
SC_HEIGHT = 800
SIDE_WIDTH = 650

# Name for the length of surface:

TABLE_SURF_HEADERS = ['Length', 'Slope', 'X0', 'Y0']
TABLE_SURF_INPUTS = {
    'Length': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 5.0,
        'measure': ' m'
    },
    'Slope': {
        'type': "float",
        'min': - 2.0,
        'max': 2.0,
        'step': 0.1,
        'init': 0.2,
        'measure': ' of Pi'
    },
    'X0': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 0.0,
        'measure': ' m'
    },
    'Y0': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 0.0,
        'measure': ' m'
    },
}
TABLE_BOX_HEADERS = ['Width', 'Height', 'X0', 'Y0']
TABLE_BOX_INPUTS = {
    'Width': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 5.0,
        'measure': ' m'
    },
    'Height': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 3.0,
        'measure': ' m'
    },
    'X0': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 3.0,
        'measure': ' m'
    },
    'Y0': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 3.0,
        'measure': ' m'
    },
}
TABLE_BODY_HEADERS = ['Body type', 'Material', 'Mass', 'Geometry_1', 'Geometry_2', 'Angle', 'X', 'Y', 'Vx', 'Vy', 'Ang_v']
TABLE_BODY_INPUTS = {
    'Body type': {
        'type': "combo",
        'data': MovableTypes.member_names
    },
    'Material': {
        'type': "combo",
        'data': Materials.member_names
    },
    'Mass': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 0.2,
        'measure': ' kG'
    },
    'Geometry_1': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.01,
        'init': 0.60,
        'measure': ' m'
    },
    'Geometry_2': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.01,
        'init': 0.2,
        'measure': ' m'
    },
    'Angle': {
        'type': "float",
        'min': - 2.0,
        'max': 2.0,
        'step': 0.1,
        'init': 0.2,
        'measure': ' of Pi'
    },
    'X': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 0.5,
        'measure': ' m'
    },
    'Y': {
        'type': "float",
        'min': 0.0,
        'max': float('inf'),
        'step': 0.1,
        'init': 0.5,
        'measure': ' m'
    },
    'Vx': {
        'type': "float",
        'min': - float('inf'),
        'max': float('inf'),
        'step': 0.1,
        'init': 0.0,
        'measure': ' m/s'
    },
    'Vy': {
        'type': "float",
        'min': - float('inf'),
        'max': float('inf'),
        'step': 0.1,
        'init': 0.0,
        'measure': ' m/s'
    },
    'Ang_v': {
        'type': "float",
        'min': - float('inf'),
        'max': float('inf'),
        'step': 0.1,
        'init': 0.0,
        'measure': ' rad/s'
    },
}
TABLE_CONTACT_HEADERS = ['Surface', 'Body', 'Point', 'Mu']


def to_pixels(input_str):
    return float(input_str) * METER


def from_pixels(pixs):
    return round(pixs/METER, 3)


class GTLPoint():
    def __init__(self, point, point_type):
        self.g_point = QtCore.QPointF()
        self.t_point = ''
        self.l_point = np.array([0., 0])


def lab_to_graph(lab_point: Union[npt.NDArray, Tuple[float, float]]) -> QtCore.QPointF:
    x_l, y_l = lab_point
    x_g = x_l * METER
    y_g = SC_HEIGHT - y_l * METER
    return QtCore.QPointF(x_g, y_g)


def graph_to_lab(graph_point: QtCore.QPointF) -> Tuple[npt.NDArray, Tuple[float, float], str]:
    x_g, y_g = graph_point.x(), graph_point.y()
    x_l = x_g / METER
    y_l = (SC_HEIGHT - y_g) / METER
    return np.array([x_l, y_l]), (x_l, y_l), lab_to_tab((x_l, y_l))


def lab_to_tab(lab_value: Union[npt.NDArray, Tuple[float, float], float]) -> str:
    if isinstance(lab_value, float):
        return f'{round(lab_value, 3)}'
    x_l, y_l = map(float, lab_value)
    return f'{round(x_l, 3)}, {round(y_l, 3)}'


def body_legend_pos(body):
    return graph_to_lab(body.anchor_pos)[2]


def input_surf_to_graph(surf_input):
    length = surf_input[TABLE_SURF_HEADERS.index('Length')]
    slope = surf_input[TABLE_SURF_HEADERS.index('Slope')]
    x_0 = surf_input[TABLE_SURF_HEADERS.index('X0')]
    y_0 = surf_input[TABLE_SURF_HEADERS.index('Y0')]
    x_1, y_1 = x_0 + length * cos(slope * pi), y_0 + length * sin(slope * pi)
    start = lab_to_graph((x_0, y_0))
    finish = lab_to_graph((x_1, y_1))
    return start, finish, to_pixels(length), slope


def input_body_to_graph(body_input):
    body_type = body_input[TABLE_BODY_HEADERS.index('Body type')]
    body_geometry = tuple(map(to_pixels, (body_input[TABLE_BODY_HEADERS.index('Geometry_1')],
                                          body_input[TABLE_BODY_HEADERS.index('Geometry_2')])))
    x0_lab = body_input[TABLE_BODY_HEADERS.index('X')]
    y0_lab = body_input[TABLE_BODY_HEADERS.index('Y')]
    angle_lab = body_input[TABLE_BODY_HEADERS.index('Angle')]
    point_0 = lab_to_graph((x0_lab, y0_lab))
    angle_graph = - angle_lab*180.0
    return body_type, body_geometry, point_0, angle_lab
