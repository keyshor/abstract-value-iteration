# Classic cart-pole system implemented by Rich Sutton et al.
# Copied from http://incompleteideas.net/sutton/book/code/pole.c
# permalink: https://perma.cc/C9ZM-652R
from spectrl.main.learning import ProductMDP, learn_policy, HyperParams
from spectrl.main.spec_compiler import ev, alw, seq
from spectrl.ars.rl import test_policy

from util.io import parse_command_line_options

from gym import spaces
from gym.utils import seeding

import math
import gym
import pickle
import numpy as np


class CartPoleEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 5
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within
        # bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.time_limit = 600
        self.time = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = action[0]*self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot *
                theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) /\
                   (self.length * (4.0/3.0 - self.masspole *
                                   costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot)
        self.time = self.time + 1
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians \
            or self.time_limit < self.time
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                print('Step after Done.')
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.time = 0
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset = cartheight/4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# Define the resource model
# Empty resource model
# sys_state: np.array(4)
# res_state: np.array(0)
# sys_action: np.array(1)
def resource_delta(sys_state, res_state, sys_action):
    return np.array([])


# Define the specification
# 1. Relevant atomic predicates:
# a. Reach predicate
#    goal: np.array(1), err: float
def reach(goal, err):
    def predicate(sys_state, res_state):
        return (min([sys_state[0] - goal,
                     goal - sys_state[0]]) + err)
    return predicate


# b. Safety Predicate
#    err: float
def balance(goal, err):
    def predicate(sys_state, res_state):
        return (min([sys_state[2] - goal,
                     goal - sys_state[2]]) + err)
    return predicate


# Stay to right of point
def stay_to_left(point):
    def predicate(sys_state, res_state):
        return point - sys_state[0]
    return predicate


# Specifications
spec = seq(alw(balance(0, 12 * 2 * math.pi / 360), ev(reach(0.5, 0.1))),
           alw(balance(0, 12 * 2 * math.pi / 360), ev(reach(0.0, 0.1))))


if __name__ == '__main__':

    flags = parse_command_line_options()
    itno = flags['itno']
    folder = flags['folder']

    # Construct Product MDP and learn policy
    system = CartPoleEnv()
    env = ProductMDP(system, 1, spec, 0.0, 1.0)
    params = HyperParams(30, 2, 8000, 40, 10, 0.04, 0.3, 0.01)

    policy, log_info = learn_policy(env, params)

    # Save policy and log information
    np.save(folder + '/log_info{}.npy'.format(itno), log_info)
    policy_file = open(folder + '/policy{}'.format(itno), 'wb')
    pickle.dump(policy, policy_file)
    policy_file.close()

    # Print rollout and performance
    _, succ_rate = test_policy(env, policy, 100)
    print('Estimated Satisfaction Rate: {}%'.format(succ_rate))
