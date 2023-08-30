"""
Классы предметов
"""

from typing import Union, Tuple, Dict, Any, List, Callable
import numpy as np
import numpy.typing as npt
from collections import namedtuple
from itertools import count
from math import cos, sin, pi, sqrt
from cmath import rect
from dataclasses import dataclass, field
from constants import *
from enum import Enum


Shape = namedtuple('Shape', ['vol', 'J'])


class MovableTypes(Enum):
    BALL = Shape(2 * pi / 3, 0.4)
    BRICK = Shape(1, None)
    RING = Shape(0, 1.0)
    WHEEL = Shape(pi, 0.5)
    NO_SPIN = Shape(2 * pi / 3, 0.0)
    POINT = Shape(0.0, 0.0)

    @classmethod
    @property
    def member_names(cls):
        return list(cls._member_names_)

    @property
    def name(self):
        return self._name_


BODY_STATE = ['position', 'angle', 'v', 'ang_v']


Mat = namedtuple("Mat", ['E', 'mu', 'rho'])


class Materials(Enum):
    WOOD = Mat(10**10, 0.5, 700)
    STEEL = Mat(2*10**11, 0.1, 7800)
    RESIN = Mat(2*10**6, 0.3, 1000)
    COMMON = Mat(2 * 10 ** 8, 0.02, 1000)

    @classmethod
    @property
    def member_names(cls):
        return list(cls._member_names_)


class V:
    def __init__(self, x: float, y: float):
        self.loc = np.array([[x], [y]])

    @property
    def len(self):
        return np.linalg.norm(self.loc)

    @property
    def dir(self):
        return self.loc/self.len

    def __add__(self, other):
        return V(*(self.loc + other.loc))


@dataclass()
class Glass:
    w: float
    h: float
    liquid: Any
    x0: float = 0.0
    y0: float = 0.0

    def set_pos(self, point: Union[npt.ArrayLike, npt.NDArray]) -> None:
        self.x0 = point[0]
        self.y0 = point[1]

    def in_area(self, point: Union[npt.ArrayLike, npt.NDArray]) -> bool:
        return self.x0 <= point[0] <= self.x0 + self.w and self.y0 <= point[1] <= self.y0 + self.h

    def add_substance(self, sub):
        self.liquid = sub


@dataclass()
class Movable:
    mass: float
    material: str
    geometry: Tuple[float, float]

    id: int = field(default_factory=count().__next__, init=False)
    name = 'Body'
    position = np.array([0., 0.])
    v = np.array([0., 0.])
    ang_v = 0.0
    p = np.array([0., 0.])
    d = 0.0
    vol = 0.0
    rho = 1000
    angle = 0.0
    path = []
    angles = []
    velocities = []
    ang_velocities = []
    accelerates = []
    ang_accelerates = []
    k_energies = []
    brut_forces = []
    potential_forces = []
    force_moments = []
    substant_forces = []
    links = []
    contact_surface = None
    suspensions = []
    connections = []
    contact_bodies = []
    collide_surface = None
    lifetime = 0.0
    timeline = []

    def __post_init__(self):
        self.mat = Materials[self.material]
        self.rho = self.mat.value.rho


    # @property
    # def name(self):
    #     return f"{self.__class__.__name__}-{self.id}"

    def set_name(self, new_name):
        self.name = new_name

    @property
    def center(self):
        return self.position

    def set_v(self, v: Union[npt.ArrayLike, npt.NDArray]) -> None:
        v_n = normalize(v)
        self.v, self.p = v_n, self.mass*v_n

    def set_ang_v(self, ang_v):
        self.ang_v = ang_v

    # def modify_p(self, dp: Union[npt.ArrayLike, npt.NDArray]) -> None:
    #     dp_n = normalize(dp)
    #     new_p = self.p + dp_n
    #     self.p = new_p
    #     self.v = new_p/self.mass

    def set_p(self, p):
        self.p = p
        self.v = p / self.mass

    def set_pos(self, pos: Union[npt.ArrayLike, npt.NDArray]) -> None:
        self.position = normalize(pos)

    def set_angle(self, phi):
        self.angle = phi

    @property
    def rotation_matrix(self):
        return np.array(
            [
                [cos(pi * self.angle), -sin(pi * self.angle)],
                [sin(pi * self.angle), cos(pi * self.angle)]
            ]
        )

    def set_collide_surface(self, collide: Union[Tuple, None] = None) -> None:
        self.collide_surface = collide

    def set_contact_surface(self, surf):
        self.contact_surface = surf

    def add_substant_force(self, force: Callable) -> None:
        self.substant_forces.append(force)

    def remove_surface(self):
        self.contact_surface = None

    def add_body(self, body):
        self.contact_bodies.append(body)

    def remove_body(self, body):
        self.contact_bodies.remove(body)

    def furthest_point(self, direction: npt.NDArray) -> npt.NDArray:
        pass

    @property
    def k_energy(self):
        return np.linalg.norm(self.v)**2/2*self.mass

    @property
    def get_state(self):
        state = [self.path[-1], self.angles[-1], self.velocities[-1], self.ang_velocities[-1]]
        return dict(zip(BODY_STATE, state))

    @property
    def volume(self) -> float:
        return 0.0

    @property
    def p_energy(self):
        return sum(p_f.energy(self) for p_f in self.potential_forces)

    @property
    def potentials_force(self):
        return sum(self.potential_forces)

    @property
    def subs_force(self):
        s = np.array([0., 0.])
        for force in self.substant_forces:
            s += force(self)
        return s

    def contacts_force(self, non_react) -> npt.NDArray:
        if self.contact_surface:
            if reaction := self.contact_surface.reaction(non_react):
                normal, tangent, moment = reaction
                self.force_moments.append(moment)
                return normal + tangent
            else:
                self.contact_surface = None
                return np.array([0, 0])
        else:
            return np.array([0, 0])

    @property
    def suspensions_force(self) -> npt.NDArray:
        return [susp.reaction() for susp in self.suspensions] if self.suspensions else np.array([0., 0.])

    @property
    def connections_force(self) -> npt.NDArray:
        return np.array([0., 0.])

    @property
    def non_reaction_forces(self):
        # print('non reactions', self.subs_force)
        return self.potentials_force + self.subs_force + self.connections_force

    @property
    def result_force(self):
        # print('result force', self.non_reaction_forces + self.contacts_force + self.suspensions_force)
        non_react_f = self.non_reaction_forces
        # print(non_react_f)
        contact_f = self.contacts_force(non_react_f)
        # print('result force', np.linalg.norm(non_react_f + contact_f))
        return non_react_f + contact_f + self.suspensions_force

    @property
    def result_force_moment(self):
        # print('result moment')
        return sum(self.force_moments)

    def accelerate(self):
        a = self.result_force / self.mass
        self.accelerates.append(a)
        # print('accelerate', a)
        return a

    def drift(self, tact):
        return self.velocities[-1] * tact + self.accelerates[-1] / 2 * tact * tact

    def reach_time(self, dr: npt.NDArray) -> float:
        d = np.linalg.norm(dr)
        # print('reach time. d', d)
        if d == 0:
            return 0
        v_proj = np.dot(self.velocities[-1], dr) / d
        a_proj = np.dot(self.accelerates[-1], dr) / d
        # print('reach time. v_proj, a_proj', v_proj, a_proj)
        if v_proj == 0:
            if a_proj == 0:
                return float('inf')
            else:
                return sqrt(2 * d / abs(a_proj)) * np.sign(a_proj)
        t_0 = d / v_proj
        # print('reach time', t_0 * (1 - a_proj * t_0))
        return t_0 * (1 - a_proj * t_0)

    def pre_step(self, tact):
        pos, vel, ang = self.position, self.v, self.angle
        self.path.append(pos)
        self.velocities.append(vel)
        self.angles.append(ang)
        a = self.accelerate()
        virt_drift = self.drift(tact)
        return virt_drift

    def one_step(self, tact: float) -> Tuple[npt.NDArray, npt.NDArray]:
        new_pos = self.position + self.drift(tact)
        self.position = new_pos
        new_p = self.p + self.accelerates[-1] * self.mass * tact
        if self.contact_surface:
            new_p = directize(new_p, self.contact_surface.surface.tau)
        self.set_p(new_p)
        return new_pos, new_p

    # def post_step(self, new_pos: npt.NDArray, new_p: npt.NDArray, tact: float):
    #     new_time = self.lifetime + tact
    #     self.lifetime = new_time
    #     self.timeline.append(new_time)
    #     self.path.append(new_pos)
    #     self.velocities.append(new_p/self.mass)
    #     self.k_energies.append(self.k_energy)
    #     # print('post step dun')


class MatPoint(Movable):
    point_id = 0

    def __init__(self, mass: float, mat, geometry):
        super(MatPoint, self).__init__(mass, mat, geometry)
        self.geometry = (0, 0)
        MatPoint.point_id += 1
        self.id = MatPoint.point_id


@dataclass()
class Brick(Movable):
    type_name: str = 'BRICK'

    def __post_init__(self):
        self.w, self.h = self.geometry
        self.mat = Materials[self.material]
        self.rho = self.mat.value.rho
        self.vol = self.mass / self.rho
        self.d = self.vol / self.w / self.h
        self.ang_velocities.append(0)

    @property
    def center(self):
        diagon = np.array([self.w / 2], [self.h / 2])
        return self.position + np.dot(self.rotation_matrix, diagon)

    def pre_step(self, tact):
        v_drift = super(Brick, self).pre_step(tact)
        k = self.k_energy
        self.k_energies.append(k)
        return v_drift


@dataclass()
class Ball(Movable):
    type_name: str = 'BALL'
    ang_v: float = 0
    colliding = 0.0

    def __post_init__(self):
        # print('post init', self.__dict__)
        self.radius = self.geometry[0]
        self.mat = Materials[self.material]
        self.rho = self.mat.value.rho
        self.body_type = MovableTypes[self.type_name]
        if self.type_name in ('BALL', 'NO_SPIN'):
            self.d = 2 * self.radius
            self.vol = self.body_type.value.vol * self.d * self.radius ** 2
            if self.mass == 0:
                self.mass = self.rho * self.vol
            else:
                self.rho = self.mass / self.vol
        elif self.type_name == 'WHEEL':
            self.vol = self.mass / self.rho
            self.d = self.vol / self.body_type.value.vol / self.radius ** 2

        self.J = self.body_type.value.J
        # print('end post_init')

    def furthest_point(self, direction: npt.NDArray) -> npt.NDArray:
        # print('far point', self.position, self.radius * direction)
        return self.position + self.radius * direction

    def set_colliding(self, timer: float = 0.0):
        self.colliding = timer

    def volume(self) -> float:
        return 4*pi*self.radius**3/3

    def set_ang_v(self, omega):
        self.ang_v = omega

    def ang_accelerate(self):
        w = self.result_force_moment / (self.J * self.mass * self.radius ** 2)
        self.ang_accelerates.append(w)
        return w

    @property
    def k_energy(self):
        return super(Ball, self).k_energy + self.J * self.mass * self.radius**2 * self.ang_v**2 / 2

    def pre_step(self, tact):
        v_drift = super(Ball, self).pre_step(tact)
        ang_vel, w = self.ang_v, self.ang_accelerate()
        k = self.k_energy
        self.k_energies.append(k)
        self.ang_velocities.append(ang_vel)
        self.force_moments = []
        return v_drift

    def one_step(self, tact: float) -> Tuple[npt.NDArray, npt.NDArray]:
        new_pos, new_p = super(Ball, self).one_step(tact)
        # print('ball step')
        if self.type_name != 'NO_SPIN':
            # print('rotation calc', self.ang_v, self.ang_accelerates[-1])
            # print('ang vels', self.ang_velocities)
            # vels_mod = [np.linalg.norm(v) / self.radius / 2 / pi for v in self.velocities]
            # print('vels', vels_mod)
            rotation = self.ang_v * tact + self.ang_accelerates[-1] / 2 * tact * tact
            # print('rotation', rotation)
            self.set_angle((self.angle - rotation / pi) % 2)
            new_ang_v = self.ang_v + self.ang_accelerates[-1] * tact
            # print('new ang_v', new_ang_v)
            # new_ang_v = new_p / self.mass / self.radius
            if self.contact_surface:
                new_ang_v = np.dot(self.v, self.contact_surface.surface.tau) / self.radius
            self.set_ang_v(new_ang_v)
        return new_pos, new_p


@dataclass()
class Surface:
    length: float
    name: str = 'the_wall'
    id: int = field(default_factory=count().__next__, init=False)
    mat: Material = common
    start: Union[npt.ArrayLike, npt.NDArray] = np.array([0., 0.])
    angle: float = 0.0
    angle_grad: float = 0.0
    slope: npt.NDArray = np.array([[1., 0.0], [0., 1.0]])
    normal: npt.NDArray = np.array([0., 1.])
    tau: npt.NDArray = np.array([1., 0.])

    @property
    def surname(self):
        return f'Surface#{self.id}'

    def set_name(self, name):
        self.name = name

    @property
    def table_view(self):
        values = (str(self.id),
                  f'Length: {self.length}',
                  f'{self.angle_grad} deg',
                  f'{self.start[0][0]}, {self.start[1][0]}'
        )
        return {k: v for k, v in zip(TABLE_SURF_COLS, values)}

    def set_pos(self, x: float, y: float) -> None:
        self.start = np.array([x, y])

    def set_slope(self, angle: float) -> None:
        self.angle = angle
        self.angle_grad = 180 * angle
        self.slope = np.array([[cos(pi * angle), -sin(pi * angle)], [sin(pi * angle), cos(pi * angle)]])
        self.normal = self.slope[:, 1]
        self.tau = self.slope[:, 0]

    def contains(self, point: Union[npt.ArrayLike, npt.NDArray]) -> bool:
        projection = np.dot(self.tau, point - self.start)
        norm = np.linalg.norm(point - self.start)
        return abs(projection - norm) <= 0.0001 and norm <= self.length * 1.0001

    def dist(self, point: npt.NDArray) -> float:
        d = np.cross(self.tau, point - self.start)
        # print('d', d)
        # if self.tau[0] != 0:
            # print((point - self.start - d * self.normal)[0])
            # l = (point - self.start - d * self.normal)[0] / self.tau[0]
            # print('l', l)
        # else:
        #     l = (point - self.start - d * self.normal)[1] / self.tau[1]
        # if 0 < l < self.length:
        #     collide_point = self.start + l * self.tau
        # elif l <= 0:
        #     collide_point = self.start
        #     d = np.linalg.norm(point - collide_point)
        # elif l >= self.length:
        #     collide_point = self.start + self.length * self.tau
        #     d = np.linalg.norm(point - collide_point)
        # print('d', d, 'tau', self.tau)
        return d

    def to_collide(self, point, drift) -> Union[bool, npt.NDArray]:
        h = self.dist(point)
        # print('to collide. point', point)
        # print('to collide h', h)
        if (normal_drift := np.dot(self.normal, drift)) >= 0 or h <= 0.0:
            return False
        # print('normal drift', normal_drift)
        # print('to collide', - h * self.normal)
        return - h * self.normal if - normal_drift >= h else False


if __name__ == "__main__":
    ball_1 = Ball(0.2, 0.05)
    ball_2 = Ball(0.3, 0.1, 'RING')
    brick = Brick(0.5, 0.1, 0.05)
    print(ball_1.J, ball_2.J)
    print(brick.__dict__)
