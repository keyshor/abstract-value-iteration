from sarl.envs.rooms import GridParams, AbstractMap, RoomsEnv
from sarl.examples.rooms_envs import GRID_PARAMS_LIST

import unittest

# parameters for a 2-by-2 grid
size = (2, 2)
edges = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 1))]
room_size = (8, 8)
wall_size = (2, 2)
vertical_door = (3, 5)
horizontal_door = (3, 5)
grid_params = GridParams(size, edges, room_size,
                         wall_size, vertical_door, horizontal_door)
abstract_map = AbstractMap(grid_params)


# class to test GridParams
class TestGridParamsMethods(unittest.TestCase):

    def test_make_abstract_state(self):
        abstract_state = grid_params.make_abstract_state(((0, 0), (0, 1)))
        self.assertTrue(abstract_state.contains(
            (4, 9)), 'center state missing')
        self.assertTrue(abstract_state.contains(
            (3, 8)), 'top-left state missing')
        self.assertTrue(abstract_state.contains(
            (5, 10)), 'bottom-right state missing')
        self.assertFalse(abstract_state.contains((2, 9)), 'incorrect state')

    def test_build_room_graph(self):
        room_graph = grid_params.build_room_graph()
        correct_graph = [[(0, 1), (1, 0)], [(0, 0), (1, 1)],
                         [(0, 0)], [(0, 1)]]
        self.check_graph_equivalence(room_graph, correct_graph)

    def test_build_abstract_graph(self):
        abstract_graph = grid_params.build_abstract_graph(
            final_state=None).graph
        correct_graph = [[1, 2], [0], [0]]
        self.check_graph_equivalence(abstract_graph, correct_graph)

    def check_graph_equivalence(self, graph1, graph2):
        self.assertEqual(len(graph1), len(graph2))
        for i in range(len(graph1)):
            self.assertEqual(len(graph1[i]), len(graph2[i]))
            for j in range(len(graph1[i])):
                self.assertTrue(graph1[i][j] in graph2[i], 'missing edges')
                self.assertTrue(graph2[i][j] in graph1[i], 'extra edges')


# class to test AbstractMap
class TestAbstractMap(unittest.TestCase):

    def test_get_abstract_state(self):
        states = [(0, 0), (4, 9), (3, 8), (5, 10), (9, 4), (8, 4), (10, 14)]
        abs_states = [-1, 0, 0, 0, 1, 1, 2]
        for i in range(len(states)):
            self.assertEqual(abstract_map.get_abstract_state(
                states[i]), abs_states[i])


# class to test RoomsEnv
class TestRoomsEnv(unittest.TestCase):

    def test_path_clear(self):
        grid_params_orig = GRID_PARAMS_LIST[0]
        start_state = grid_params_orig.make_abstract_state(
            grid_params_orig.edges[0], is_start=True)
        goal_state = grid_params_orig.make_abstract_state(
            grid_params_orig.edges[-1])
        env = RoomsEnv(
            GRID_PARAMS_LIST[0], start_state, goal_state, goal_state, max_timesteps=250)
        self.assertTrue(env.path_clear((4, 4), (5, 5)))
        self.assertTrue(env.path_clear((4, 8.5), (5, 7.5)))
        self.assertTrue(env.path_clear((7.5, 4), (8.5, 4)))
        self.assertTrue(env.path_clear((14, 18.5), (14, 18.1)))
        self.assertTrue(env.path_clear((9.88, 4.07), (10.04, 4.2)))
        self.assertFalse(env.path_clear((4.9, 8.5), (5.9, 7.5)))
        self.assertFalse(env.path_clear((2, 7.5), (2, 8.5)))
