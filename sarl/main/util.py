import gym
import numpy as np


# wrap gym env with additional state components from goal abstract state
class GoalWrapper(gym.Env):

    def __init__(self, wrapped_env, goal_abstract_state, start_abstract_state=None):
        '''
        wrapped_env: Gym environment
        goal_abstract_state: Abstract state with an "addition_state" method
        start_abstract_state: Abstract state with an "additional_state" method
        '''
        self._wrapped_env = wrapped_env
        self._goal = goal_abstract_state
        self._start = start_abstract_state

        self._extra_dim = np.shape(
            self._goal.additional_state(self._goal.sample()))[0]
        if self._start is not None:
            self._extra_dim *= 2
        self._obs_dim = self._wrapped_env.observation_space.shape[0]
        self._shape = (self._obs_dim + self._extra_dim,)

    def _get_full_state(self, state):
        full_state = np.concatenate([state, self._goal.additional_state(state)])
        if self._start is not None:
            full_state = np.concatenate([full_state, self._start.additional_state(state)])
        return full_state

    def reset(self):
        return self._get_full_state(self._wrapped_env.reset())

    def step(self, action):
        obs, rew, done, info = self._wrapped_env.step(action)
        return self._get_full_state(obs), rew, done, info

    def render(self):
        self._wrapped_env.render()

    def cum_reward(self, sarss):
        return self._wrapped_env.cum_reward(sarss)

    def get_extra_dim(self):
        return self._extra_dim

    @property
    def observation_space(self):
        high = np.inf * np.ones(self._shape)
        low = -high
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def action_space(self):
        return self._wrapped_env.action_space


# wrap policy for goal env to get basic policy
class GoalPolicy:

    def __init__(self, policy, goal_abstract_state, start_abstract_state=None):
        self._policy = policy
        self._goal = goal_abstract_state
        self._start = start_abstract_state

    def _get_full_state(self, state):
        full_state = np.concatenate([state, self._goal.additional_state(state)])
        if self._start is not None:
            full_state = np.concatenate([full_state, self._start.additional_state(state)])
        return full_state

    def get_action(self, state):
        return self._policy.get_action(self._get_full_state(state))


# combines all training envs into a single env with random selection
class MultiRandomEnv(gym.Env):

    def __init__(self, envs):
        self.envs = envs
        self.curr = np.random.randint(len(self.envs))

    def reset(self):
        self.curr = np.random.randint(len(self.envs))
        return self.envs[self.curr].reset()

    def step(self, action):
        obs, rew, done, info = self.envs[self.curr].step(action)
        return obs, rew, done, info

    def render(self):
        self.envs[self.curr].render()

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space
