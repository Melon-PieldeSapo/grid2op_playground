import random
import warnings
import numpy as np
import copy

from grid2op.Opponent import BaseOpponent
from grid2op.Exceptions import OpponentError


class MyRandomOpponent(BaseOpponent):
    """
    An opponent that disconnect at random any powerlines among a specified list given
    at the initialization.

    """

    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self._do_nothing = None
        self._line_attacks = None
        self._generators_attacks = None
        self._lines_ids = None
        self._generators_ids = None

    def init(self, partial_env, lines_attacked=[], generators_attacked=[], **kwargs):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Used when the opponent is created.

        Parameters
        ----------
        partial_env
        lines_attacked
        kwargs

        Returns
        -------

        """
        # this if the function used to properly set the object.
        # It has the generic signature above,
        # and it's way more flexible that the other one.

        if len(generators_attacked) == 0:
            warnings.warn(f'The opponent is deactivated as there is no information as to which generator to attack. '
                          f'You can set the argument "kwargs_opponent" to the list of the line names you want '
                          f' the opponent to attack in the "make" function.')

        # Store attackable GENERATOR IDs
        self._generators_ids = []
        for g_name in generators_attacked:
            g_id = np.where(self.action_space.name_gen == g_name)
            if len(g_id) and len(g_id[0]):
                self._generators_ids.append(g_id[0][0])
            else:
                raise OpponentError("Unable to find the generator named \"{}\" on the grid. For "
                                    "information, generators on the grid are : {}"
                                    "".format(g_name, sorted(self.action_space.name_gen)))

        # Pre-build attacks actions
        self._generators_attacks = []
        for g_id in self._generators_ids:
            att = self.action_space({'redispatch': [(g_id, -50)]})
            self._generators_attacks.append(att)
        self._generators_attacks = np.array(self._generators_attacks)

        if len(lines_attacked) == 0:
            warnings.warn(f'The opponent is deactivated as there is no information as to which line to attack. '
                          f'You can set the argument "kwargs_opponent" to the list of the line names you want '
                          f' the opponent to attack in the "make" function.')

        # Store attackable lines IDs
        self._lines_ids = []
        for l_name in lines_attacked:
            l_id = np.where(self.action_space.name_line == l_name)
            if len(l_id) and len(l_id[0]):
                self._lines_ids.append(l_id[0][0])
            else:
                raise OpponentError("Unable to find the powerline named \"{}\" on the grid. For "
                                    "information, powerlines on the grid are : {}"
                                    "".format(l_name, sorted(self.action_space.name_line)))

        # Pre-build attacks actions
        self._line_attacks = []
        for l_id in self._lines_ids:
            att = self.action_space({'set_line_status': [(l_id, -1)]})
            self._line_attacks.append(att)
        self._line_attacks = np.array(self._line_attacks)

    def attack(self, observation, agent_action, env_action,
               budget, previous_fails):
        if observation is None:  # during creation of the environment
            return None, 0  # i choose not to attack in this case
        if random.random() < 0.5:#atack Powerline
            # Status of attackable lines
            status = observation.line_status[self._lines_ids]
            # If all attackable lines are disconnected
            if np.all(~status):
                return None, 0  # i choose not to attack in this case

            # Pick a line among the connected lines
            attack = self.space_prng.choice(self._line_attacks[status])
            # print(F"attack: {attack}")
            return attack, None
        else:
            # Pick a generator
            attack = self.space_prng.choice(self._generators_attacks)
            return attack, None

    def _custom_deepcopy_for_copy(self, new_obj, dict_=None):
        super()._custom_deepcopy_for_copy(new_obj, dict_)
        if dict_ is None:
            dict_ = {}

        new_obj._line_attacks = copy.deepcopy(self._line_attacks)
        new_obj._lines_ids = copy.deepcopy(self._lines_ids)
