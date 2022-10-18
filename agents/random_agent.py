from envs.swat_gym import SWATEnv

class RandomAgent(object):
    """
    This baseline agent selects random actions at each time step
    """
    def __init__(self, start_date=None):
        self.start_date = start_date
    
    def select_action(self, env):
        action = env.action_space.sample() 
        return action
