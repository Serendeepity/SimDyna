from collections import namedtuple
import numpy as np
import numpy.typing as npt


Material = namedtuple("Mat", ['E', 'mu'])
common = Material(9*10**8, 0.002)
wood = Material(15.0, 0.5)

MATS = {
    'common': common,
    'wood': wood,
}
TABLE_SURF_COLS = ['id', 'Length', 'Slope', "Start Point"]
TABLE_BODY_COLS = ['id', 'Mass', 'Geometry', 'Angle', "Position", 'Velocity', 'K energy']


def normalize(point):

    if isinstance(point, np.ndarray):
        if point.shape == (2,):
            return point
        else:
            return point.reshape(2,)
    else:
        return np.array([point[0], point[1]])


def directize(vector, direction):
    return np.dot(vector, direction) * direction
