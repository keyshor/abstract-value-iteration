from util.rl import test_policy, get_rollout

import numpy as np

# global constants
R_MAX = 1000000.


# class representing a semi-markov decision process
# expected reward and probabilities are estimated from sample runs
class SMDP:

    # abstract_map: implements get_abstract_state(state) - maps concrete state to abstract state
    # options: [[Option]] list of options available at each abstract state
    # env_generator: maps a state to an MDP with original rewards but modified
    #                start state, ends in any abstract state other that the starting abstract state
    # gamma: discount factor
    def __init__(self, abstract_map, options, env_generator, gamma, num_rollouts=100):
        self.abstract_map = abstract_map
        self.options = options.copy()
        self.options.append([])
        self.num_states = len(self.options)
        self.env_generator = env_generator
        self.gamma = gamma
        self.rewards, steps_reward = self.estimate_rewards(num_rollouts)
        self.probs, steps_prob = self.estimate_probs(num_rollouts)
        self.steps_taken = steps_reward + steps_prob

    # estimate reward from samples
    def estimate_rewards(self, num_rollouts):
        rewards = []
        num_steps = 0
        for q1_options in self.options:
            q1_rewards = []
            for option in q1_options:
                reward = 0
                for _ in range(num_rollouts):
                    start_state = option.sample_start_state()
                    env = self.env_generator(start_state)
                    cur_reward, steps_taken = test_policy(env, option.policy, 1,
                                                          self.gamma, get_steps=True)
                    reward += cur_reward
                    num_steps += steps_taken
                q1_rewards.append(reward / num_rollouts)
            rewards.append(q1_rewards)
        return rewards, num_steps

    # estimate discounted probabilities from samples
    def estimate_probs(self, num_rollouts):
        probs = []
        num_steps = 0
        for q1_options in self.options:
            q1_probs = []
            for option in q1_options:
                prob_vector = np.zeros(self.num_states)
                for _ in range(num_rollouts):
                    start_state = option.sample_start_state()
                    env = self.env_generator(start_state)
                    sarss = get_rollout(env, option.policy, False)
                    num_steps += len(sarss)
                    q = self.abstract_map.get_abstract_state(sarss[-1][-1])
                    prob_vector[q] += self.gamma ** len(sarss)
                prob_vector = prob_vector / num_rollouts
                q1_probs.append(prob_vector)
            probs.append(q1_probs)
        return probs, num_steps

    def pretty_print(self):
        print('\n**** SMDP ****')
        print('Rewards: {}'.format(self.rewards))
        print('Discounted Probabilities:')
        for prob_vectors in self.probs:
            print(np.array(prob_vectors).tolist())
