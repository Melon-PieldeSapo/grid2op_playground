# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Action.PlayableAction import PlayableAction


class PowerlineSetAction(PlayableAction):
    """
    This type of :class:`PlayableAction` only implements the modifications 
    of the grid through "set_line_status" keyword.

    Nothing else is supported and any attempt to use something 
    else will have no impact.

    """
    authorized_keys = {
        "set_line_status"
    }
    attr_list_vect = [
        "_set_line_status"
    ]
    attr_list_set = set(attr_list_vect)

    def __init__(self):
        super().__init__()
