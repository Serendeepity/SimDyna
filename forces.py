from typing import Union, Tuple, Dict, Any, List, Callable
import numpy as np
import numpy.typing as npt
from math import cos, sin, pi, sqrt
from objects import Movable, Surface, Ball
from abc import ABC, abstractmethod
from dataclasses import dataclass
from constants import *
from enum import Enum

G = 9.8
GROUND_ZERO = 0.0
K_W = 1.5


class PreSubs(Enum):
    VACUUM = 0.0, 0.0
    AIR = 1.0, 0.000018
    WATER = 1000.0, 0.001

    @classmethod
    @property
    def member_names(cls):
        return list(cls._member_names_)


class Mats(Enum):
    common = 9 * 10 ** 8, 0.02
    wood = 15.0, 0.5


@dataclass()
class Force:
    def __init__(self, value: npt.ArrayLike, target: Movable):
        self.value = np.array(value)
        self.target = target

    def __add__(self, other):
        if self.target == other.target:
            return Force(self.value + other.value, self.target)
        else:
            return None


@dataclass()
class Substance:
    rho: float = 0.0
    """
    Eta - коэффициент вязкости.
    Воздух 1.8 * 10**(-5) Па*с
    Вода 10**(-3)
    
    """
    eta: float = 0.0
    g: npt.NDArray = 9.8 * np.array([0, -1])

    def set_gravity(self, g: npt.NDArray):
        self.g = g

    def friction_maker(self) -> Callable:

        if self.eta == 0.0:
            return lambda x: np.array([0, 0])

        def friction(body: Movable):
            v_mod = np.linalg.norm(body.v)
            Re = body.d * self.rho * v_mod/self.eta
            return - 3.0*pi * body.d * body.v if Re < 300.0 else - pi * body.d**2 * v_mod * body.v / 16.0

        return friction

    def archimedes_maker(self) -> Callable:

        def archimedes(body):
            return - self.rho * self.g * body.vol

        return archimedes


@dataclass()
class PotentialField(ABC):
    const: float

    @abstractmethod
    def potential(self, point: Union[npt.ArrayLike, npt.NDArray]) -> float:
        pass

    @abstractmethod
    def intensity(self, point: Union[npt.ArrayLike, npt.NDArray]) -> npt.NDArray:
        pass

    @abstractmethod
    def force(self, body: Movable) -> npt.NDArray:
        pass

    @abstractmethod
    def energy(self, body: Movable) -> float:
        pass


@dataclass()
class GForce(PotentialField):
    const = G

    def potential(self, point: Union[npt.ArrayLike, npt.NDArray]) -> float:
        return G*(point[1] - GROUND_ZERO)

    def intensity(self, point: Union[npt.ArrayLike, npt.NDArray]) -> npt.NDArray:
        return G*np.array([0.0, -1.0])

    def force(self, body: Movable) -> npt.NDArray:
        return body.mass*self.intensity(body.position)

    def energy(self, body: Movable) -> float:
        return body.mass*self.potential(body.position)


@dataclass()
class ContactBrickS:
    surface: Surface
    body: Movable
    contact_point: float
    mu: float

    def reaction(self):
        impact_t, impact_n = np.dot(self.surface.slope.T, self.body.non_reaction_forces)
        return - self.surface.normal * impact_n[0], \
               - self.surface.tau * min(- self.mu * impact_n[0], abs(impact_t[0])) * np.sign(impact_t[0])

    def moment(self):
        return 0.0


@dataclass()
class ContactBallS:
    surface: Surface
    ball: Ball
    contact_point: float
    mu: float

    def set_contact_point(self, new_point):
        self.contact_point = new_point

    def reaction(self, non_reaction_forces):
        # print('contact check reaction', self.surface.dist(self.ball.furthest_point(- self.surface.normal)))
        if not self.surface.contains(self.ball.furthest_point(- self.surface.normal)):
            return None
        impact_t, impact_n = np.dot(self.surface.slope.T, non_reaction_forces.reshape(2, 1))
        shoulder = self.surface.normal * self.ball.radius
        # print('impacts', impact_t, impact_n)
        # print('non reaction forces', non_reaction_forces)
        n_reaction, tau_reaction = - impact_n, - impact_t * self.ball.J / (1 + self.ball.J)
        # print('n, tau', - impact_n, - impact_t * self.ball.J / (1 + self.ball.J))
        # print(shoulder, tau_reaction)
        moment_reaction = np.cross(shoulder, tau_reaction * self.surface.tau)
        # print('reactions', n_reaction * self.surface.normal, tau_reaction * self.surface.tau, moment_reaction)
        return self.surface.normal * n_reaction, self.surface.tau * tau_reaction, moment_reaction

@dataclass()
class CollideBS:
    ball: Ball
    surface: Surface
    recovery: float
    spin: bool

    def __post_init__(self):
        e_ball = self.ball.mat.value.E
        e_surf = self.surface.mat.E
        self.k = pi * e_ball * e_surf / (e_ball + e_surf)
        self.collide_time = sqrt(self.ball.mass/self.ball.radius/self.k)
        self.mu = min(self.ball.mat.value.mu, self.surface.mat.mu)

    # def delta_p(self):
    #     p_tau_0, p_norm_0 = np.dot(self.surface.slope.T, self.ball.p)
    #     print('p norm', p_norm_0)
    #     if p_norm_0 < - 0.01:
    #         n_delta_p = -(self.recovery + 1) * p_norm_0
    #     else:
    #         n_delta_p = - p_norm_0
    #     print(n_delta_p)
    #     # tau_delta_p = - min(self.mu * n_delta_p, abs(p_tau_0))
    #     tau_delta_p = 0
    #     return n_delta_p * self.surface.normal + tau_delta_p * self.surface.tau

    def after_collide(self) -> Tuple:
        p_tau_0, p_norm_0 = np.dot(self.surface.slope.T, self.ball.p.reshape(2, 1))
        # if (new_p_norm := - self.recovery * p_norm_0) < 0.005:
        #     new_p_norm = 0
        #     rebound = False
        # else:
        #     rebound = True
        new_p_norm = - self.recovery * p_norm_0
        new_v_tau = (self.ball.ang_v * self.ball.J * self.ball.radius + p_tau_0 / self.ball.mass) / (1 + self.ball.J)
        new_pos = self.ball.position + (p_tau_0 / self.ball.mass + new_v_tau) * self.surface.tau * self.collide_time / 2
        # print('after collide. pos', new_pos)
        new_pos_correct = new_pos - (self.ball.radius - self.surface.dist(new_pos)) * self.surface.normal
        # print('after collide. correct pos', new_pos_correct)
        new_ang_v = new_v_tau / self.ball.radius * (self.ball.J != 0)
        new_ang = self.ball.angle + (self.ball.ang_v + new_ang_v) * self.collide_time / 2
        # print('collider.after collide. new p_norm', new_p_norm)
        new_p = (self.surface.normal * new_p_norm, self.surface.tau * new_v_tau * self.ball.mass)
        return new_p, new_pos_correct, new_ang_v, new_ang


vacuum = Substance(0.0, 0.0, np.array([0.0, 0.0]))
air = Substance(1.0, 0.000018)
water = Substance(1000.0, 0.001)

SUBS = {
    'vacuum': vacuum,
    'air': air,
    'water': water
}
