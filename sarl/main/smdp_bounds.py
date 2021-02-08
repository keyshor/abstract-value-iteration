from util.rl import test_policy, get_rollout

import numpy as np

# global constants
R_MAX = 1000000.


# class representing a semi-markov decision process
# reward and probability bounds are estimated from sample runs
class SMDP:

    # abstract_map: implements get_abstract_state(state) - maps concrete state to abstract state
    # options: [[Option]] list of options available at each abstract state
    # env_generator: maps a state to an MDP with original rewards but modified
    #                start state, ends in any abstract state other that the starting abstract state
    # gamma: discount factor
    def __init__(self, abstract_map, options, env_generator, gamma,
                 num_samples=10, num_rollouts=10):
        self.abstract_map = abstract_map
        self.options = options.copy()
        self.options.append([])
        self.num_states = len(self.options)
        self.env_generator = env_generator
        self.gamma = gamma
        self.reward_min, self.reward_max, steps_reward = self.estimate_reward_bounds(
            num_samples, num_rollouts)
        self.prob_min, self.prob_max, steps_prob = self.estimate_prob_bounds(
            num_samples, num_rollouts)
        self.num_steps = (steps_prob + steps_reward)

    # estimate min and max reward from samples
    def estimate_reward_bounds(self, num_samples, num_rollouts):
        r_min = []
        r_max = []
        num_steps = 0
        for q1_options in self.options:
            r_min_q1 = []
            r_max_q1 = []
            for option in q1_options:
                r_min_q1_q2 = R_MAX
                r_max_q1_q2 = -R_MAX
                for _ in range(num_samples):
                    start_state = option.sample_start_state()
                    env = self.env_generator(start_state)
                    reward, steps = test_policy(
                        env, option.policy, num_rollouts, self.gamma, get_steps=True)
                    num_steps += steps
                    r_min_q1_q2 = min(r_min_q1_q2, reward)
                    r_max_q1_q2 = max(r_max_q1_q2, reward)
                r_min_q1.append(r_min_q1_q2)
                r_max_q1.append(r_max_q1_q2)
            r_min.append(r_min_q1)
            r_max.append(r_max_q1)
        return r_min, r_max, num_steps

    # estimate min and max discounted probabilities from samples
    def estimate_prob_bounds(self, num_samples, num_rollouts):
        p_min = []
        p_max = []
        num_steps = 0
        for q1_options in self.options:
            p_min_q1 = []
            p_max_q1 = []
            for i in range(len(q1_options)):
                option = q1_options[i]
                p_min_q1_q2 = np.ones(self.num_states) * R_MAX
                p_max_q1_q2 = -np.ones(self.num_states) * R_MAX
                for _ in range(num_samples):
                    start_state = option.sample_start_state()
                    env = self.env_generator(start_state)
                    probs = np.zeros(self.num_states)
                    for _ in range(num_rollouts):
                        sarss = get_rollout(env, option.policy, False)
                        num_steps += len(sarss)
                        q = self.abstract_map.get_abstract_state(sarss[-1][-1])
                        probs[q] += self.gamma ** len(sarss)
                    probs = probs / num_rollouts
                    p_min_q1_q2 = np.minimum(p_min_q1_q2, probs)
                    p_max_q1_q2 = np.maximum(p_max_q1_q2, probs)
                p_min_q1.append(p_min_q1_q2)
                p_max_q1.append(p_max_q1_q2)
            p_min.append(p_min_q1)
            p_max.append(p_max_q1)
        return p_min, p_max, num_steps

    def pretty_print(self):
        print('\n**** SMDP ****')
        print('Min Rewards: {}'.format(self.reward_min))
        print('Max Rewards: {}'.format(self.reward_max))
        print('Min Discounted Probabilities:')
        for probs in self.prob_min:
            print(np.array(probs).tolist())
        print('Max Discounted Probabilities:')
        for probs in self.prob_max:
            print(np.array(probs).tolist())
