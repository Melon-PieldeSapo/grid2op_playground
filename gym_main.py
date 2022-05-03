import grid2op
from grid2op.gym_compat import GymEnv

import grid2op
env_name = ...
env = grid2op.make(env_name)

from grid2op.gym_compat import GymEnv
# this of course will not work... Replace "AGymSpace" with a normal gym space, like Dict, Box, MultiDiscrete etc.
from gym.spaces import AGymSpace
gym_env = GymEnv(env)

class MyCustomObservationSpace(AGymSpace):
    def __init__(self, whatever, you, want):
        # do as you please here
        pass
        # don't forget to initialize the base class
        AGymSpace.__init__(self, see, gym, doc, as, to, how, to, initialize, it)
        # eg. Box.__init__(self, low=..., high=..., dtype=float)

    def to_gym(self, observation):
        # this is this very same function that you need to implement
        # it should have this exact name, take only one observation (grid2op) as input
        # and return a gym object that belong to your space "AGymSpace"
        return SomethingThatBelongTo_AGymSpace
        # eg. return np.concatenate((obs.gen_p * 0.1, np.sqrt(obs.load_p))

gym_env.observation_space = MyCustomObservationSpace(whatever, you, wanted)

env_name = "l2rpn_case14_sandbox"  # or any other grid2op environment name
g2op_env = grid2op.make(env_name)  # create the gri2op environment

gym_env = GymEnv(g2op_env)  # create the gym environment

# check that this is a properly defined gym environment:
import gym
print(f"Is gym_env and open AI gym environment: {isinstance(gym_env, gym.Env)}")
# it shows "Is gym_env and open AI gym environment: True"