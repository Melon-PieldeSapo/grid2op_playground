
from grid2op.Opponent import RandomLineOpponent
import warnings
import numpy as np
import copy

from grid2op.Opponent import BaseOpponent
from grid2op.Exceptions import OpponentError

class MyRandomLineOpponent(RandomLineOpponent):

    def attack(self, observation, agent_action, env_action,
               budget, previous_fails):
        if observation is None:  # during creation of the environment
            return None, 0  # i choose not to attack in this case

        # Status of attackable lines
        status = observation.line_status[self._lines_ids]
        # If all attackable lines are disconnected
        if np.all(~status):
            return None, 0  # i choose not to attack in this case

        # Pick a line among the connected lines
        attack = self.space_prng.choice(self._attacks[status])
        #print(F"attack: {attack}")
        return attack, None
