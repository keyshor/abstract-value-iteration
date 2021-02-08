# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Adapted from rllab maze_env_utils.py.
   Modified it by considering abstract state and abstract map"""
import numpy as np
import math

from numpy import linalg as LA


# AntMazeAbstarctState class representing a rectangular region for AntMaze environment
class AntMazeAbstractState:

    # region: [(x1,y1),(x2,y2)] - [left-down, right-up]
    def __init__(self, index, sample_center_only=True):
        if index == 0:
            # center = (0., 0.)
            region = [(-1.0, -1.0), (1.0, 1.0)]
        elif index == 1:
            # center = (14., 2.)
            region = [(13.0, 1.0), (15.0, 3.0)]
        elif index == 2:
            # center = (14., 14.)
            region = [(13.0, 13.0), (15.0, 15.0)]
        elif index == 3:
            # center = (0., 16.0)
            region = [(-1.0, 15.0), (1.0, 17.0)]
        else:
            raise Exception('Check the index for this abstract state!')
        self.region = np.array(region)
        self.center = (self.region[0] + self.region[1]) / 2
        self.size = self.region[1] - self.region[0]
        self.sample_center_only = sample_center_only
        if index == 0:
            self.sample_center_only = True

    # s: np.array(2) or array-like
    def contains(self, s):
        return s[0] >= self.region[0][0] and s[0] <= self.region[1][0] \
            and s[1] >= self.region[0][1] and s[1] <= self.region[1][1]

    # sample a point from the region
    def sample(self):
        if self.sample_center_only:
            return self.center + np.random.uniform(size=self.center.size, low=-.1, high=.1)
        else:
            return self.center + np.random.uniform(size=self.center.size, low=-1., high=1.)

    # using robustness semantics for reward
    # s : np.array(2)
    def reward(self, s):
        return -LA.norm(s[:2] - self.center)

    # additional state needed for shared policy
    def additional_state(self, s):
        return self.center - s[:2]


# AntPushAbstarctState class representing a rectangular region and
# a box's position (up-left) for AntPush environment
class AntPushAbstractState:

    # region: [(x1,y1),(x2,y2)] - [left-down, right-up]
    # box_center: (xb1, yb1) - box's center
    # region and box_region are in ant's coordinate
    def __init__(self, index, sample_center_only=True):
        if index == 0:
            region = [(-0.5, -0.5), (0.5, 0.5)]
            box_pos = (0., 0.)
        elif index == 1:
            region = [(-10., -0.5), (-9., 0.5)]
            box_pos = (0., 0.)
        elif index == 2:
            region = [(-10., 9.), (-9., 10.)]
            box_pos = (0., 0.)
        elif index == 3:
            region = [(-2.1, 9.), (-1.9, 10.)]
            box_pos = [(1.9, 0.), (2.1, 0.)]
        elif index == 4:
            region = [(-0.5, 18.5), (0.5, 19.5)]
            box_pos = (2.1, 0.)
        else:
            raise Exception('Check the index for this abstract state!')
        self.region = np.array(region)
        self.box_pos = np.array(box_pos)
        self.center = (self.region[0] + self.region[1]) / 2
        self.size = (self.region[1] - self.region[0]) / 2
        self.sample_center_only = sample_center_only
        if index == 0:
            self.sample_center_only = True
        if self.sample_center_only:
            self.size = np.array([0.1, 0.1])

    # s: np.array(2) or array-like
    def contains(self, s):
        return s[0] >= self.region[0][0] and s[0] <= self.region[1][0] \
            and s[1] >= self.region[0][1] and s[1] <= self.region[1][1]

    # sample a point from the region
    # check the sampled position does not collide with the box
    def sample(self):
        random_sample = np.random.uniform(
            size=self.center.size, low=-self.size, high=self.size)
        if len(self.box_pos.shape) > 1:
            box_center = (self.box_pos[0] + self.box_pos[1]) / 2
            box_random_sample = np.random.uniform(
                size=self.center.size, low=[random_sample[0], 0.], high=[self.size[0], 0.])
            return np.concatenate([random_sample + self.center, box_random_sample + box_center])
        else:
            return np.concatenate([random_sample + self.center, self.box_pos])

    def reward(self, s):
        box_penalty = 0.
        if s[16] > 0.01:
            box_penalty = -10
        return -LA.norm(s[:2] - self.center) + box_penalty

    def additional_state(self, s):
        return self.center - s[:2]


# AntFallAbstarctState class representing a rectangular region and
# a box's position (up-left) for AntFall environment
class AntFallAbstractState:

    # region: [(x1,y1),(x2,y2)] - [left-down, right-up]
    def __init__(self, index, sample_center_only=True):
        ant_height = []
        if index == 0:
            region = [(-0.5, -0.5), (0.5, 0.5)]
            box_pos = (0., 0.)
        elif index == 1:
            region = [(7.5, 2.), (8.5, 3.)]
            box_pos = (0., 0.)
        elif index == 2:
            region = [(7.5, 10.5), (8.5, 11.5)]
            box_pos = [(7., 0.), [8., 0.]]
        elif index == 3:
            region = [(7.5, 22.5), (8.5, 23.5)]
            box_pos = (8., 0.)
        elif index == 4:
            region = [(-0.5, 26.5), (0.5, 27.5)]
            box_pos = (8., 0.)
        else:
            raise Exception('Check the index for this abstract state!')
        self.region = np.array(region)
        self.box_pos = np.array(box_pos)
        self.ant_height = np.array(ant_height)
        self.center = (self.region[0] + self.region[1]) / 2
        self.size = (self.region[1] - self.region[0]) / 2
        self.sample_center_only = sample_center_only
        if index == 0:
            self.sample_center_only = True
        if self.sample_center_only:
            self.size = np.array([0.1, 0.1])

    # s: np.array(2) or array-like
    def contains(self, s):
        return s[0] >= self.region[0][0] and s[0] <= self.region[1][0] \
            and s[1] >= self.region[0][1] and s[1] <= self.region[1][1]

    # sample a point from the region
    # check the sampled position does not collide with the box
    def sample(self):
        random_sample = np.random.uniform(
            size=self.center.size, low=-self.size, high=self.size)
        if len(self.box_pos.shape) > 1:
            box_center = (self.box_pos[0] + self.box_pos[1])/2
            box_random_sample = np.random.uniform(
                size=self.center.size, low=[random_sample[1], 0.], high=[self.size[1], 0.])
            box_loc = box_center + box_random_sample
        else:
            box_loc = self.box_pos
        return np.concatenate([random_sample + self.center, box_loc, self.ant_height])

    # s : np.array(2)
    def reward(self, s):
        return -LA.norm(s[:2] - self.center)

    def additional_state(self, s):
        return self.center - s[:2]

# function for building abstract graph for the environment
# env_id: string ('Maze', 'Push' or 'Fall')


def build_abstract_graph(env_id, sample_center_only=True):
    abstract_states = []
    if env_id == 'Maze':
        for index in range(4):
            abstract_state = AntMazeAbstractState(index, sample_center_only)
            abstract_states.append(abstract_state)
    elif env_id == 'Push':
        for index in range(5):
            abstract_state = AntPushAbstractState(index, sample_center_only)
            abstract_states.append(abstract_state)
    elif env_id == 'Fall':
        for index in range(5):
            abstract_state = AntFallAbstractState(index, sample_center_only)
            abstract_states.append(abstract_state)
    else:
        raise Exception('Invalid ENV_ID: {}'.format(env_id))

    return MazeAbstractGraph(env_id, abstract_states)


# class implementing a map from concrete state to abstract state
# can also be used as a final region
class AbstractMap:
    def __init__(self, abstract_states):
        self.abstract_states = abstract_states

    def get_abstract_state(self, state):
        s = -1
        for i in range(len(self.abstract_states)):
            if self.abstract_states[i].contains(state):
                s = i
                break
        return s

    def contains(self, state):
        return self.get_abstract_state(state) != -1


class MazeAbstractGraph:

    def __init__(self, env_id, abstract_states):
        self.env_id = env_id
        self.abstract_states = abstract_states
        if self.env_id == 'Maze':
            self.graph = [[1], [2], [3], []]
        elif self.env_id == 'Push':
            self.graph = self.graph = [[1, 2], [2, 3], [3], [4], []]
        elif self.env_id == 'Fall':
            self.graph = [[1], [2], [3], [4], []]
        else:
            raise Exception('env_id is not in the current environment list')

    def pretty_print(self):
        print('\n**** Abstract Graph ****')
        print('Adjacency List:', self.graph)


class Move(object):
    X = 11
    Y = 12
    Z = 13
    XY = 14
    XZ = 15
    YZ = 16
    XYZ = 17
    SpinXY = 18


def can_move_x(movable):
    return movable in [Move.X, Move.XY, Move.XZ, Move.XYZ,
                       Move.SpinXY]


def can_move_y(movable):
    return movable in [Move.Y, Move.XY, Move.YZ, Move.XYZ,
                       Move.SpinXY]


def can_move_z(movable):
    return movable in [Move.Z, Move.XZ, Move.YZ, Move.XYZ]


def can_spin(movable):
    return movable in [Move.SpinXY]


def can_move(movable):
    return can_move_x(movable) or can_move_y(movable) or can_move_z(movable)


def construct_maze(maze_id='Maze'):
    if maze_id == 'Maze':
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 'Push':
        structure = [
            [1, 1,  1,  1,   1],
            [1, 0, 'r', 1,   1],
            [1, 0,  Move.XY, 0,  1],
            [1, 1,  0,  1,   1],
            [1, 1,  1,  1,   1],
        ]
    elif maze_id == 'Fall':
        structure = [
            [1, 1,   1,  1],
            [1, 'r', 0,  1],
            [1, 0,   Move.YZ,  1],
            [1, -1, -1,  1],
            [1, 0,   0,  1],
            [1, 1,   1,  1],
        ]
    elif maze_id == 'Block':
        O = 'r'
        structure = [
            [1, 1, 1, 1, 1],
            [1, O, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 'BlockMaze':
        O = 'r'
        structure = [
            [1, 1, 1, 1],
            [1, O, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
    else:
        raise NotImplementedError(
            'The provided MazeId %s is not recognized' % maze_id)

    return structure


def line_intersect(pt1, pt2, ptA, ptB):
    """
    Taken from https://www.cs.hmc.edu/ACM/lectures/intersections.html

    this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE:
        return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)


def ray_segment_intersect(ray, segment):
    """
    Check if the ray originated from (x, y) with direction theta intersects
    the line segment (x1, y1) -- (x2, y2),
    and return the intersection point if there is one
    """
    (x, y), theta = ray
    # (x1, y1), (x2, y2) = segment
    pt1 = (x, y)
    len = 1
    pt2 = (x + len * math.cos(theta), y + len * math.sin(theta))
    xo, yo, valid, r, s = line_intersect(pt1, pt2, *segment)
    if valid and r >= 0 and 0 <= s <= 1:
        return (xo, yo)
    return None


def point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
