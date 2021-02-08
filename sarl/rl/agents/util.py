from tf_agents.trajectories import time_step as ts


class GymPyPolicy:
    '''
    Wrapper to convert a tf agents py_policy to a numpy policy
    '''

    def __init__(self, py_policy):
        self._policy = py_policy

    def get_action(self, obs, policy_state=()):
        time_step = ts.restart(obs)
        action_step = self._policy.action(time_step, policy_state=policy_state)
        return action_step.action
