import grid2op
import torch
from grid2op.Agent import RandomAgent
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeReplay

import grid2op
from grid2op.Runner import Runner
from grid2op.Agent import RandomAgent

from pytorch_DQN_test import DqnGrid2op

env_name = "rte_case14_realistic"  # for example, other environments might be usable
env = grid2op.make()
NB_EPISODE = 10  # assess the performance for 10 episodes, for example
NB_CORE = 1  # do it on 2 cores, for example

PATH_SAVE = "./data/"
# create an environment and an agent
env = grid2op.make(env_name)
agent = DqnGrid2op(env, seed=10)
agent.qnetwork_local.load_state_dict(torch.load('1651596314.4605389/4000checkpoint.pth'))

# create a runner
runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)

# run and save the results
res = runner.run(nb_episode=NB_EPISODE, nb_process=NB_CORE, path_save=PATH_SAVE)

# and now load it and play the "movie"
plot_epi = EpisodeReplay(PATH_SAVE)
print(res)
for epi in range(NB_EPISODE):
    plot_epi.replay_episode(res[epi][1], gif_name="this_episode.gif")
