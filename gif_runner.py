import grid2op
from grid2op.Agent import RandomAgent
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeReplay

path_saved_data = "./data/"
# create an environment and an agent
env = grid2op.make()
agent = RandomAgent(env.action_space)

# create a runner
runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)

# run and save the results
res = runner.run(nb_episode=1, path_save=path_saved_data)

# and now load it and play the "movie"
plot_epi = EpisodeReplay(path_saved_data)
plot_epi.replay_episode(res[0][1], gif_name="this_episode.gif")