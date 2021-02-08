import numpy as np
import gym
import math

from numpy import linalg as LA


# AbstarctState class representing a rectangular region
class AbstractState:

    # region: [(x1,y1),(x2,y2)]
    def __init__(self, region, is_start=False):
        self.region = np.array(region)
        self.is_start = is_start
        self.size = self.region[1] - self.region[0]
        self.center = (self.region[1] + self.region[0]) / 2

    # s: np.array(2) or array-like
    def contains(self, s):
        return s[0] >= self.region[0][0] and s[0] <= self.region[1][0] \
            and s[1] >= self.region[0][1] and s[1] <= self.region[1][1]

    # sample a point from the region
    def sample(self):
        if self.is_start:
            return (np.random.random_sample(2) * self.size/2) + self.region[0] + (self.size/4)
        else:
            return np.random.random_sample(2) * self.size + self.region[0]

    # using l2 reward
    # s : np.array(2)
    def reward(self, s):
        return -LA.norm(s - self.center)


# class representing a list of abstract states and their connections
class AbstractGraph:

    # abstract_states: [AbstractState] (AbstractState implements contains(state): Bool and
    #                                   sample(): np.array(state_dim))
    # graph: adjacency list of possible directed edges denoting nearby abstract states
    def __init__(self, abstract_states, graph):
        self.abstract_states = abstract_states
        self.graph = graph

    def pretty_print(self):
        print('\n**** Abstract Graph ****')
        print('Adjacency List: {}'.format(self.graph))

    def compute_indegrees(self):
        indegrees = [0] * len(self.abstract_states)
        for out_edges in self.graph:
            for vertex in out_edges:
                indegrees[vertex] += 1
        return indegrees


# parameters for defining the rooms environment
class GridParams:

    # size: (h:int, w:int) specifying size of grid
    # edges: list of pairs of adjacent rooms (room is a pair (x,y) - 0 based indexing)
    #        first coordinate is the vertical position (just like matrix indexing)
    # room_size: (l:int, b:int) size of a single room (height first)
    # wall_size: (tx:int, ty:int) thickness of walls (thickness of horizontal wall first)
    # vertical_door, horizontal_door: relative coordinates for door, specifies min and max
    #                                 coordinates for door space
    def __init__(self, size, edges, room_size, wall_size, vertical_door, horizontal_door):
        self.size = np.array(size)
        self.edges = edges
        self.room_size = np.array(room_size)
        self.wall_size = np.array(wall_size)
        self.partition_size = self.room_size + self.wall_size
        self.vdoor = np.array(vertical_door)
        self.hdoor = np.array(horizontal_door)
        self.graph = self.make_adjacency_matrix()
        self.grid_region = AbstractState([np.array([0., 0.]), self.size * self.partition_size])

    # map a room to an integer
    def get_index(self, r):
        return self.size[1]*r[0] + r[1]

    # make abstract state for the given edge
    def make_abstract_state(self, edge, is_start=True):
        r1, r2 = edge
        if not (r1[0] <= r2[0] and r1[1] <= r2[1]):
            # swap if r1 is not smaller
            r1, r2 = r2, r1
        x = r1[0]*self.partition_size[0]
        y = r1[1]*self.partition_size[1]
        if r1[0] == r2[0] and r1[1] == r2[1] - 1:
            # r2 is to the right of r1
            x1 = x + self.vdoor[0]
            y1 = y + self.room_size[1]
            x2 = x + self.vdoor[1]
            y2 = y + self.partition_size[1]
            return AbstractState([(x1, y1), (x2, y2)], is_start)
        elif r1[0] == r2[0] - 1 and r1[1] == r2[1]:
            # r2 is below r1
            x1 = x + self.room_size[0]
            y1 = y + self.hdoor[0]
            x2 = x + self.partition_size[0]
            y2 = y + self.hdoor[1]
            return AbstractState([(x1, y1), (x2, y2)], is_start)
        else:
            raise Exception('Edge between rooms that are not adjacent given: ' + (r1, r2))

    # build graph (adjacency list) of how rooms are connected
    def build_room_graph(self):
        num_nodes = self.size[0]*self.size[1]
        graph = [[] for _ in range(num_nodes)]
        for r1, r2 in self.edges:
            graph[self.get_index(r1)].append(r2)
            graph[self.get_index(r2)].append(r1)
        return graph

    # build abstract graph from grid definition
    # no outgoing edges from final_state
    def build_abstract_graph(self, final_state=None):
        # Step 1: contruct a list of abstract states
        abstract_states = []
        (r1_start, r2_start) = self.edges[0]
        abstract_states.append(self.make_abstract_state((r1_start, r2_start), is_start=True))
        for r1, r2 in self.edges[1:]:
            abstract_states.append(self.make_abstract_state((r1, r2)))

        # Step 2: construct room graph
        graph = self.build_room_graph()

        # Step 3: create index_map
        index_map = self.create_index_map()

        # Step 4: construct abstract state graph (line graph of room graph)
        line_graph = []
        for r1, r2 in self.edges:
            line_edges = []
            for ra, rb in [(r1, r2), (r2, r1)]:
                for r in graph[self.get_index(ra)]:
                    if not r == rb:
                        line_edges.append(index_map[(ra, r)])
            line_graph.append(line_edges)

        if final_state is not None:
            line_graph[final_state] = []

        return AbstractGraph(abstract_states, line_graph)

    # create a map from edges to position in the edge list
    def create_index_map(self):
        index_map = {}
        index = 0
        for r1, r2 in self.edges:
            index_map[(r1, r2)] = index
            index_map[(r2, r1)] = index
            index += 1
        return index_map

    # takes pairs of adjacent rooms and creates a h*w-by-4 matrix of booleans
    # returns the compact adjacency matrix
    def make_adjacency_matrix(self):
        graph = [[False]*4 for _ in range(self.size[0]*self.size[1])]
        for r1, r2 in self.edges:
            graph[self.get_index(r1)][self.get_direction(r1, r2)] = True
            graph[self.get_index(r2)][self.get_direction(r2, r1)] = True
        return graph

    # returns the direction of r2 from r1
    def get_direction(self, r1, r2):
        if r1[0] == r2[0]+1 and r1[1] == r2[1]:
            return 0  # up
        elif r1[0] == r2[0] and r1[1] == r2[1]+1:
            return 1  # left
        elif r1[0] == r2[0]-1 and r1[1] == r2[1]:
            return 2  # down
        elif r1[0] == r2[0] and r1[1] == r2[1]-1:
            return 3  # right
        else:
            raise Exception('Given rooms are not adjacent!')


# class implementing a map from concrete state to abstract state
# can also be used as a final region
class AbstractMap:

    def __init__(self, grid_params):
        self.grid_params = grid_params
        self.index_map = grid_params.create_index_map()

    # assumes state is a valid state
    def get_abstract_state(self, state):
        r = (state//self.grid_params.partition_size).astype(np.int)
        pos = state - (r * self.grid_params.partition_size)
        abs_state = None

        # Case 1: horizontal door
        if (pos[0] >= self.grid_params.room_size[0]
            and pos[1] >= self.grid_params.hdoor[0]
                and pos[1] <= self.grid_params.hdoor[1]):
            abs_state = ((r[0], r[1]), (r[0]+1, r[1]))

        # Case 2: vertical door
        elif (pos[1] >= self.grid_params.room_size[1]
              and pos[0] >= self.grid_params.vdoor[0]
                and pos[0] <= self.grid_params.vdoor[1]):
            abs_state = ((r[0], r[1]), (r[0], r[1]+1))

        # Case 3: state is at top most line in room
        elif (pos[0] == 0
              and pos[1] >= self.grid_params.hdoor[0]
                and pos[1] <= self.grid_params.hdoor[1]):
            abs_state = ((r[0], r[1]), (r[0]-1, r[1]))

        # Case 4: state is at left most line in room
        elif (pos[1] == 0
              and pos[0] >= self.grid_params.vdoor[0]
                and pos[0] <= self.grid_params.vdoor[1]):
            abs_state = ((r[0], r[1]), (r[0], r[1]-1))

        if abs_state in self.index_map:
            return self.index_map[abs_state]
        else:
            return -1

    def contains(self, state):
        return self.get_abstract_state(state) != -1


# return discounted cumulative reward for the given rollout
def discounted_reward(sarss, gamma):
    sarss_rev = sarss.copy()
    sarss_rev.reverse()
    reward = 0.0
    for _, _, r, _ in sarss_rev:
        reward = r + gamma*reward
    return reward


# Environment modelling 2d grid with rooms
class RoomsEnv(gym.Env):

    # grid_params: GridParams
    # start_region: AbstractState
    # final_region: (for deciding when done
    #               enough if this implements contains(state))
    # goal_region: AbstractState (for rewards, should implement reward(state))
    def __init__(self, grid_params, start_region, final_region, goal_region,
                 max_timesteps=20, start_state=None, use_shaped_rewards=False):
        self.grid_params = grid_params
        self.start_region = start_region
        self.final_region = final_region
        self.goal_region = goal_region
        self.max_timesteps = max_timesteps
        self.start_state = start_state
        self.use_shaped_rewards = use_shaped_rewards
        # set the initial state
        self.steps = 0
        self.state = self.start_state
        if self.state is None:
            self.state = self.start_region.sample()

    def reset(self):
        self.steps = 0
        self.state = self.start_state
        if self.state is None:
            self.state = self.start_region.sample()
        return self.state

    # action: np.array(2)
    def step(self, action):
        action = np.array([action[0] * math.cos(action[1]),
                           action[0] * math.sin(action[1])])
        old_state = self.state
        new_state = self.state + action
        self.steps += 1
        if self.path_clear(old_state, new_state):
            self.state = new_state
        reward = self.goal_region.reward(self.state)
        return self.state, reward, self.steps > self.max_timesteps, None

    @property
    def observation_space(self):
        shape = self.state.shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        max_vel = np.amin(self.grid_params.wall_size) / 2
        high = np.array([max_vel, math.pi / 2])
        low = -high
        return gym.spaces.Box(low, high)

    def render(self):
        print(self.state.tolist())

    def cum_reward(self, sarss):
        return (max([self.goal_region.reward(s) for s, _, _, _ in sarss])
                + discounted_reward(sarss, gamma=0.99))

    # Check if straight line joining s1 and s2 does not pass through walls
    # s1 is assumed to be a legal state
    # we are assuming that wall size exceeds maximum action size
    # also assuming that door regions are small compared to rooms
    def path_clear(self, s1, s2):
        params = self.grid_params

        # find rooms of the states
        r1 = (s1//params.partition_size).astype(np.int)
        r2 = (s2//params.partition_size).astype(np.int)

        # find relative positions within rooms
        p1 = s1 - (r1 * params.partition_size)
        p2 = s2 - (r2 * params.partition_size)

        if not self.is_state_legal(s2, r2, p2):
            return False

        # both states are inside the same room (not in the door area)
        if (p1[0] <= params.room_size[0] and p1[1] <= params.room_size[1]
                and p2[0] <= params.room_size[0] and p2[1] <= params.room_size[1]):
            return True
        # both states in door area
        if ((p1[0] > params.room_size[0] or p1[1] > params.room_size[1])
                and (p2[0] > params.room_size[0] or p2[1] > params.room_size[1])):
            return True

        # swap to make sure s1 is in the room and s2 is in the door area
        if (p2[0] <= params.room_size[0] and p2[1] <= params.room_size[1]):
            p1, p2 = p2, p1
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        # four cases to consider
        if p2[0] > params.room_size[0]:
            # s1 is above s2
            if (r1 == r2).all():
                return self.check_vertical_intersect(p1, p2, params.room_size[0])
            # s1 is below s2
            else:
                return self.check_vertical_intersect((s1[0], p1[1]), (s2[0], p2[1]),
                                                     (r2[0]+1) * params.partition_size[0])
        else:
            # s1 is left of s2
            if (r1 == r2).all():
                return self.check_horizontal_intersect(p1, p2, params.room_size[1])
            # s1 is right of s2
            else:
                return self.check_horizontal_intersect((p1[0], s1[1]), (p2[0], s2[1]),
                                                       (r2[1]+1) * params.partition_size[1])

    # check if the state s is a legal state that is within the grid and not inside any wall area
    # r is the room of the state
    # p is the relative position within the room
    def is_state_legal(self, s, r, p):
        params = self.grid_params

        # make sure state is within the grid
        if not params.grid_region.contains(s):
            return False
        if r[0] >= params.size[0] or r[1] >= params.size[1]:
            return False

        # make sure state is not inside any wall area
        if (p[0] <= params.room_size[0] and p[1] <= params.room_size[1]):
            return True
        elif (p[0] > params.room_size[0] and p[1] >= params.hdoor[0]
              and p[1] <= params.hdoor[1]):
            return params.graph[params.get_index(r)][2]
        elif (p[1] > params.room_size[1] and p[0] >= params.vdoor[0]
              and p[0] <= params.vdoor[1]):
            return params.graph[params.get_index(r)][3]
        else:
            return False

    # check if line from s1 to s2 intersects the horizontal axis at a point inside door region
    # horizontal coordinates should be relative positions within rooms
    def check_vertical_intersect(self, s1, s2, x):
        y = ((s2[1] - s1[1]) * (x - s1[0]) / (s2[0] - s1[0])) + s1[1]
        return (self.grid_params.hdoor[0] <= y
                and y <= self.grid_params.hdoor[1])

    # check if line from s1 to s2 intersects the vertical axis at a point inside door region
    # vertical coordinates should be relative positions within rooms
    def check_horizontal_intersect(self, s1, s2, y):
        x = ((s2[0] - s1[0]) * (y - s1[1]) / (s2[1] - s1[1])) + s1[0]
        return (self.grid_params.vdoor[0] <= x
                and x <= self.grid_params.vdoor[1])
