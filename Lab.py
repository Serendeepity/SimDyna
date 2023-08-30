import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import numpy.typing as npt
from objects import Movable, Ball, Surface, Glass
from forces import CollideBS, PotentialField, Substance, SUBS, PreSubs, ContactBrickS, ContactBallS
from constants import *
from math import pi, sqrt


speed_params = {
    'min': 100,
    'max': 1000,
    'step': 100,
    'init': 200,
}

substance_params = PreSubs.member_names

LAB_PARAMS = [speed_params, substance_params]
LAB_INPUTS = {
    'Speed': {
        'type': "int",
        'min': 100,
        'max': 1000,
        'step': 100,
        'init': 200,
        'measure': ' tacts/second'
    },
    'Gravity': {
        'type': "check",
        'init': True
    },
    'Substance': {
        'type': "combo",
        'data': PreSubs.member_names
    },
    'k': {
        'type': 'float',
        'min': 0.0,
        'max': 1.0,
        'step': 0.1,
        'init': 1.0,
        'measure': ''
    },
    'Spin': {
        'type': "check",
        'init': False
    },
}
LAB_HEADERS = [key for key in LAB_INPUTS.keys()]

PER_SECOND = 100
GROUND_ZERO = 0.0
FORCES = []
COLLIDES = []
BODIES = []
SURFACES = []


@dataclass()
class LabSystem:
    speed: int = 100  # tacts per second
    gravity: bool = True
    pre_substance: PreSubs = PreSubs.VACUUM
    k: float = 1.0
    spin: bool = False
    G = np.array([0.0, 0.0])
    GROUND_ZERO = 0.0
    substance = SUBS['vacuum']
    surfaces = []
    bodies = []
    bindings = []
    potential_fields = []
    collides = {}
    contacts = []
    time = 0


    def __post_init__(self):
        if self.gravity:
            self.set_gravity()
        self.add_substance(self.pre_substance)

    def set_speed(self, s: int) -> None:
        self.speed = s

    def set_gravity(self, ground: float = 0.0, g: float = 9.8) -> None:
        self.G = g * np.array([0.0, -1.0])
        self.GROUND_ZERO = ground

    def add_potential(self, field: PotentialField) -> None:
        self.potential_fields.append(field)

    def add_substance(self, sub: Union[Substance, str, PreSubs]) -> None:
        if isinstance(sub, Substance):
            self.substance = sub
        elif isinstance(sub, str) and sub in SUBS.keys():
            self.substance = SUBS[sub]
        elif isinstance(sub, PreSubs):
            self.substance = Substance(*sub.value, self.G)

    def add_body(self, body: Movable,
                 start_position: Union[npt.ArrayLike, npt.NDArray, None] = None,
                 start_v: Union[npt.ArrayLike, npt.NDArray, None] = None
                 ) -> None:
        if start_position:
            body.set_pos(normalize(start_position))
        if start_v:
            body.set_v(normalize(start_v))
        # print('add body', body.name)
        body.add_substant_force(lambda x: x.mass*self.G)
        body.add_substant_force(self.substance.friction_maker())
        body.add_substant_force(self.substance.archimedes_maker())
        body.timeline.append(0)
        body.path.append(body.position)
        self.bodies.append(body)

    def add_surface(self,
                    surf: Surface,
                    start: npt.ArrayLike = (0, 0),
                    angle: float = 0.0) -> None:
        # print('check surf start', all(surf.start == np.array([0., 0.])))
        if all(surf.start == np.array([0., 0.])):
            surf.set_pos(*start)
        if surf.angle == 0.0:
            surf.set_slope(angle)
        self.surfaces.append(surf)

    def add_contact(self, params):
        # print(params)
        contact_body = [body for body in self.bodies if body.name == params['Body']][0]
        # print(contact_body)
        contact_surface = [surf for surf in self.surfaces if surf.name == params['Surface']][0]
        # print(contact_surface)
        # print('center', contact_body.center)
        real_dist = contact_surface.dist(contact_body.center)
        # print('real_dist', real_dist)
        if contact_body.type_name == 'BRICK':
            right_dist = contact_body.h / 2
        elif contact_body.type_name in ('BALL', 'RING', 'WHEEL', 'NO_SPIN'):
            right_dist = contact_body.radius
        else:
            right_dist = 0
        # print('right dist', right_dist)
        # print('new pos', contact_body.position - (real_dist - right_dist) * contact_surface.normal)
        contact_body.set_pos(contact_body.position - (real_dist - right_dist) * contact_surface.normal)
        # self.contacts.append(ContactBS(contact_surface, contact_body, params['Point'], params['Mu']))
        new_cont = ContactBallS(contact_surface, contact_body, params['Point'], params['Mu'])
        contact_body.set_contact_surface(new_cont)

    def check_ball_collide(self, body: Ball, drift: npt.NDArray) -> Union[Tuple, None]:
        print('check ball collide')
        if body.collide_surface: return None
        for surf in self.surfaces:
            if body.contact_surface and body.contact_surface.surface == surf:
                continue
            f_point = body.furthest_point(- surf.normal)
            distance = surf.to_collide(f_point, drift)
            if isinstance(distance, np.ndarray):
                # print('check ball collide. distance-2', distance)
                reach_time = body.reach_time(distance)
                collide_point = f_point + body.drift(reach_time)
                # print(reach_time, collide_point)
                if surf.contains(collide_point):
                    return reach_time, CollideBS(body, surf, self.k, self.spin), collide_point
            # if surf.contains(collide_point):
            #     if 0 < reach_time < 1 / self.speed:
            #         # print('reach time', reach_time)
            #         return reach_time, CollideBS(body, surf, self.k, self.spin)
            #     elif - 1 / self.speed < reach_time <= 0:
            #         # print(reach_time, np.linalg.norm(collide_point - surf.start))
            #         # print(ContactBallS(surf, body, np.linalg.norm(collide_point - surf.start) ,0))
            #         return reach_time, ContactBallS(surf, body, np.linalg.norm(collide_point - surf.start), 0), surf.normal * v_norm
        print('check ball collide. return None')
        return None

    # def check_ball_surf(self, body: Ball) -> None:
    #     for surf in self.surfaces:
    #         v_norm = np.dot(surf.normal.reshape(1, 2), body.v)[0]
    #         distance = surf.dist(body.far_point(-surf.normal))
    #         # print(surf.name, v_norm, distance)
    #         # print('criteria', - distance / v_norm)
    #         if body.name in self.collides.keys():
    #             if v_norm > 0 and distance > 0:
    #                 print(self.collides[body.name].sum_f)
    #                 del self.collides[body.name]
    #                 body.set_collide_force(None)
    #         else:
    #             if v_norm < 0:
    #                 if - distance / v_norm < 1 / self.speed:
    #                     new_collide = CollideBS(body, surf)
    #                     self.collides[body.name] = new_collide
    #                     # print(new_collide.force())
    #                     # print(new_collide.__dict__)
    #                     body.set_collide_force(new_collide.force())

    # def colliding_bs(self, collide: Tuple):
    #     pass
    #     # dt = collide.lasting/200
    #     # print('colliding, dt =', dt)
    #     # times = int(1//(dt*self.speed))
    #     # pos, imp = collide.ball.position, collide.ball.p
    #     # for i in range(times):
    #     #     pos, imp = collide.ball.one_step(dt)
    #     #     mom_f = collide.force()
    #     #     # print('force', mom_f)
    #     #     collide.add_f(mom_f)
    #     #     collide.ball.set_collide_force(mom_f)
    #
    #     collide.ball.post_step(pos, imp)

    def progress_ball(self, ball: Ball) -> None:
        print('progress ball')
        potential_drift = ball.pre_step(1 / self.speed)

        if dtau := ball.colliding:
            if dtau <= 1 / self.speed:
                dt = 1 / self.speed - dtau
                # ball.set_collide_surface()
                ball.set_colliding()
            else:
                dt = 0
                ball.set_colliding(dtau - 1 / self.speed)
        else:
            collide_data = self.check_ball_collide(ball, potential_drift)
            print('progress ball. collide data', collide_data)
            if collide_data:
                # if ball.collide_surface is None:
                #     dt = 1 / self.speed
                # if dtau := ball.colliding:
                #     # print('dtau', dtau)
                #     if dtau <= 1 / self.speed:
                #         dt = 1 / self.speed - dtau
                #         ball.set_collide_surface()
                #         ball.set_colliding()
                #     else:
                #         dt = 0
                #         ball.set_colliding(dtau - 1 / self.speed)
                # else:
                spurt, collider = collide_data[:-1]
                _pos, _p = ball.one_step(spurt)
                p_new, pos_new, ang_v_new, ang_new = collider.after_collide()
                # print('progress ball. after collide data', p_new, pos_new, ang_v_new, ang_new)
                # print('progress ball. ball position', ball.position)
                # print('progress ball. new normal p', np.linalg.norm(p_new[0]))
                if np.linalg.norm(p_new[0]) <= 0.05:
                    p_new = (np.array([0, 0]), p_new[1])
                    new_cont = ContactBallS(collider.surface, ball, collide_data[-1] + pos_new - ball.position, 0)
                    print('progress  ball. new contact', new_cont)
                    ball.set_contact_surface(new_cont)
                ball.set_p(sum(p_new))
                ball.set_ang_v(ang_v_new)
                # ball.set_pos(pos_new)
                ball.set_angle(ang_new)
                dtau = collider.collide_time + spurt - 1 / self.speed
                if dtau > 0:
                    dt = 0
                    ball.set_colliding(dtau)
                else:
                    dt = - dtau
                    ball.set_colliding()
                    # ball.set_collide_surface()
            else:
                dt = 1 / self.speed
                # print('progress ball. no collide')
        ball.one_step(dt)
        # print('progress ball. contact surface', ball.contact_surface)
        # print('progress ball. check contact. v', ball.velocities)
        if ball.contact_surface and np.dot(ball.v, ball.contact_surface.surface.normal) > 0.001:
            ball.contact_surface = None

    def step_body(self):
        new_states = {}
        # print('step ball')
        for ball in self.bodies:
            if ball.type_name in ('BALL', 'RING', 'WHEEL', 'NO_SPIN'):
                # if co_surface := self.check_ball_collide(ball):
                #     if type(co_surface[1]) == CollideBS:
                #         ball.set_collide_surface(co_surface)
                #     elif type(co_surface[1]) == ContactBallS:
                #         ball.add_contact_surface(co_surface[1])
                #         ball.set_pos(ball.position + ball.v * co_surface[0])
                #         correct_v = co_surface[2]
                #         ball.set_v(ball.v - correct_v)
                self.progress_ball(ball)
                # print(ball.get_state)
                new_states[ball.name] = ball.get_state
        for brick in self.bodies:
            if brick.type_name == 'BRICK':
                self.progress_brick(brick)
                new_states[brick.name] = brick.get_state
        return new_states

    def progress_brick(self, brick):
        # print('progress brick')
        brick.post_step(*brick.one_step(1 / self.speed), self.speed)


if __name__ == "__main__":
    # ball = Ball(0.5, 0.05)
    # ball_1 = Ball(0.5, 0.05)
    # print(ball, ball_1)
    # ball.set_pos((5, 0.5))
    # ball.set_v((1, 5))
    # BODIES.append(ball)
    # bottom = Surface(10.0, 'ground')
    # bottom.set_pos(0, 0.2)
    # bottom.set_slope(0.0)
    # r_wall = Surface(5, 'right wall')
    # r_wall.set_pos(5.1, 0)
    # r_wall.set_slope(0.5)
    # lid = Surface(10, 'ceiling')
    # lid.set_pos(10, 1)
    # lid.set_slope(1.0)
    # l_wall = Surface(5, 'left wall')
    # l_wall.set_pos(4.9, 0)
    # l_wall.set_slope(1.5)
    # SURFACES.extend([r_wall, bottom, l_wall])


    # _, ax = plt.subplots(figsize=(12, 12))
    # x_1, y_1 = list(zip(*ball.path))
    # x_2, y_2 = list(zip(*ball_2.path))
    # ax.plot(x_1, y_1, 'bo')
    # ax.plot(x_2, y_2, 'ro')
    # plt.show()

    lab = LabSystem(200)
    lab.set_gravity()
    lab.add_substance(PreSubs.AIR)

    print(lab.substance)
    print(PreSubs._member_names_)
    print(PreSubs.member_names)
    print(PreSubs['AIR'].value)