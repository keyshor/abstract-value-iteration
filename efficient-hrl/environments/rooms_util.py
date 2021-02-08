from environments.rooms_envs import GRID_PARAMS_LIST


# Return a function for deciding if a run is successful
def get_success_fn(rooms_id):
    params = GRID_PARAMS_LIST[rooms_id]
    goal_state = params.make_abstract_state(params.edges[-1])

    def success_fn(states):
        for state in states:
            if goal_state.contains(state):
                return True
        return False
    return success_fn
