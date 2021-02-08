import numpy as np


# value iteration for smdp
# smdp: SMDP
# num_iters: int
# return value function v
def value_iteration(smdp, num_iters, verbose=False, V_MIN=-1e24):
    # Step 1: initialization
    v = np.zeros(smdp.num_states)

    # Step 2: iteration
    for _ in range(num_iters):
        # placeholder for updated values
        v_new = V_MIN * np.ones(smdp.num_states)

        # compute new values based on old values
        for i in range(smdp.num_states):
            num_options = len(smdp.options[i])
            if num_options == 0:
                v_new[i] = 0
            for j in range(num_options):
                # updates
                v_new[i] = max(v_new[i], smdp.rewards[i][j] +
                               np.dot(smdp.probs[i][j], v))

        # store new values
        v = v_new

        if verbose:
            print('V: {}'.format(v.tolist()))

    return v


# compute a high level policy (maps abstract states to actions) from value function
# values: np.array(num_states)
# rewards: [[int]] reward for each option
# probs: [[np.array(num_states)]] prob vector for each options
def compute_high_level_policy(values, rewards, probs, V_MIN=-1e24):
    # placeholder for policy map
    num_states = len(values)
    policy = [0] * num_states
    best_values = V_MIN * np.ones(num_states)

    # find the best option from each state
    for i in range(num_states):
        for j in range(len(rewards[i])):
            value = rewards[i][j] + np.dot(probs[i][j], values)
            if value > best_values[i]:
                policy[i] = j
                best_values[i] = value

    return policy


# pretty print values
def print_values(v):
    print('\n**** Values ****')
    print('Values: {}'.format(v.tolist()))
