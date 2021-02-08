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

from sarl.envs.maze_env import MazeEnv
from sarl.envs.ant import AntEnv


class AntMazeEnv(MazeEnv):
    # def __init__(self, )
    MODEL_CLASS = AntEnv

# import os

# def main():

#   env = AntMazeEnv()
#   #import pdb; pdb.set_trace()
#   print("Testing", env.__class__)
#   #ob_space = env.observation_space
#   act_space = env.action_space
#   ob = env.reset()

#   for _ in range(1000):
#     env.render()
#     a = act_space.sample()
#   #assert act_space.contains(a)
#     res = env.step(a)
#   #assert ob_space.contains(res.observation)
#   #assert np.isscalar(res.reward)
#   #if 'CIRCLECI' in os.environ:
#   #  print("Skipping rendering test")
#   #else
#     #import pdb; pdb.set_trace()
#     #if res.done:
#     #  ob = env.reset()
#   #env.terminate()
#   #import pdb; pdb.set_trace()

# if __name__ == "__main__":
#   main()
