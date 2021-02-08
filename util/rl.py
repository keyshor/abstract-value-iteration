import numpy as np


# Policy that takes random actions
class RandomPolicy:

    # action_dim : int
    # action_bound : float (bound on absolute value of each component)
    def __init__(self, action_dim, action_bound):
        self.action_dim = action_dim
        self.action_bound = action_bound

    def get_action(self, state):
        return ((np.random.random_sample((self.action_dim,)) * self.action_bound * 2)
                - self.action_bound)


# Compute a single rollout.
#
# env: Environment
# policy: Policy
# render: bool
# return: [(np.array([state_dim]), np.array([action_dim]), float, np.array([state_dim]))]
#          ((state, action, reward, next_state) tuples)
def get_rollout(env, policy, render, max_timesteps=10000):
    # Step 1: Initialization
    state = env.reset()
    done = False

    # Step 2: Compute rollout
    sarss = []
    steps = 0
    while (not done) and (steps < max_timesteps):
        # Step 2a: Render environment
        if render:
            env.render()

        # Step 2b: Action
        action = policy.get_action(state)

        # Step 2c: Transition environment
        next_state, reward, done, _ = env.step(action)

        # Step 2d: Rollout (s, a, r)
        sarss.append((state, action, reward, next_state))

        # Step 2e: Update state
        state = next_state
        steps += 1

    # Step 3: Render final state
    if render:
        env.render()

    return sarss


# return discounted cumulative reward for the given rollout
def discounted_reward(sarss, gamma):
    sarss_rev = sarss.copy()
    sarss_rev.reverse()
    reward = 0.0
    for _, _, r, _ in sarss_rev:
        reward = r + gamma*reward
    return reward


# Estimate the cumulative reward of the policy.
#
# env: Environment
# policy: Policy
# n_rollouts: int
# return: float
def test_policy(env, policy, n_rollouts, gamma=1, use_cum_reward=False,
                get_steps=False, max_timesteps=10000):
    cum_reward = 0.0
    num_steps = 0
    for _ in range(n_rollouts):
        sarss = get_rollout(env, policy, False, max_timesteps=max_timesteps)
        num_steps += len(sarss)
        if use_cum_reward:
            cum_reward += env.cum_reward(sarss)
        else:
            cum_reward += discounted_reward(sarss, gamma)
    if get_steps:
        return cum_reward / n_rollouts, num_steps
    return cum_reward / n_rollouts


# Estimate the probability of reaching the goal.
# Works only for 0-1 rewards.
#
# env: Environment
# policy: Policy
# n_rollouts: int
# return: float
def get_reach_prob(env, policy, n_rollouts, max_timesteps=10000):
    succesful_trials = 0
    for _ in range(n_rollouts):
        sarss = get_rollout(env, policy, False, max_timesteps=max_timesteps)
        if discounted_reward(sarss, 1) > 0:
            succesful_trials += 1
    return (succesful_trials * 100) / n_rollouts


# print reward and reaching probability
def print_performance(environment, policy, gamma, n_rollouts=100, max_timesteps=10000):
    reward = test_policy(environment, policy, n_rollouts,
                         gamma=gamma, max_timesteps=max_timesteps)
    reach_prob = get_reach_prob(
        environment, policy, n_rollouts, max_timesteps=max_timesteps)
    print('\nEstimated Reward: {}'.format(reward))
    print('Estimated Reaching Probability: {}'.format(reach_prob))
    return reward, reach_prob


# print rollout
def print_rollout(env, policy, state_dim=-1):
    sarss = get_rollout(env, policy, False)
    for s, _, _, _ in sarss:
        print(s.tolist()[:state_dim])
    print(sarss[-1][-1].tolist()[:state_dim])


# log data to file
#
# file: file handle
# iter: int (iteration number)
# num_transitions: int (number of simulation steps in each iter)
# reward: float
# prob: float (satisfaction probability)
# additional_data: string
def log_to_file(file, iter, num_transitions, reward, reach_prob, additional_data={}):
    file.write('**** Iteration Number {} ****\n'.format(iter))
    file.write('Environment Steps Taken: {}\n'.format(num_transitions))
    file.write('Reward: {}\n'.format(reward))
    file.write('Satisfaction Probability: {}\n'.format(reach_prob))
    for key in additional_data:
        file.write('{}: {}\n'.format(key, additional_data[key]))
    file.write('\n')
    file.flush()
