import numpy as np


# value iteration for smdp with reward and probability bounds
# smdp: SMDP
# num_iters: int
# return v_min and v_max, bounds on the value function
def value_iteration(smdp, num_iters, verbose=False, V_MIN=-1e24):
    # Step 1: initialization
    v_min = np.zeros(smdp.num_states)
    v_max = np.zeros(smdp.num_states)

    # Step 2: iteration
    for _ in range(num_iters):
        # placeholder for updated values
        v_min_new = V_MIN * np.ones(smdp.num_states)
        v_max_new = V_MIN * np.ones(smdp.num_states)

        # compute signs of v_min and v_max
        s_min = _sign(v_min)
        s_max = _sign(v_max)

        # compute new values based on old values
        for i in range(smdp.num_states):
            num_options = len(smdp.options[i])
            if num_options == 0:
                v_min_new[i] = 0
                v_max_new[i] = 0
            for j in range(num_options):
                # compute the probability values to use in updates
                prob_min = (
                    s_min * smdp.prob_min[i][j]) + ((1-s_min) * smdp.prob_max[i][j])
                prob_max = (
                    (1-s_max) * smdp.prob_min[i][j]) + (s_max * smdp.prob_max[i][j])

                # updates
                v_min_new[i] = max(
                    v_min_new[i], smdp.reward_min[i][j] + np.dot(prob_min, v_min))
                v_max_new[i] = max(
                    v_max_new[i], smdp.reward_max[i][j] + np.dot(prob_max, v_max))

        # store new values
        v_min = v_min_new
        v_max = v_max_new

        if verbose:
            print('V_min: {}'.format(v_min.tolist()))
            print('V_max: {}'.format(v_max.tolist()))

    return v_min, v_max


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
def print_values(v_min, v_max):
    print('\n**** Values ****')
    print('Min Values: {}'.format(v_min.tolist()))
    print('Max Values: {}'.format(v_max.tolist()))


# returns a vector denoting sign of values
def _sign(values):
    v = values.copy()
    v[v > 0] = 1
    v[v <= 0] = 0
    return v
