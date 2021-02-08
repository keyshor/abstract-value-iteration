from sarl.rl.agents.td3 import TD3Params


# HyperParameter class for StableBaselines TD3
class SB_TD3Params(TD3Params):
    WRAPPED_CLS = TD3Params


# Policy that wraps stable baselines model
class SBPolicy:

    def __init__(self, model):
        self.model = model

    def get_action(self, obs):
        action, _ = self.model.predict(obs)
        return action
