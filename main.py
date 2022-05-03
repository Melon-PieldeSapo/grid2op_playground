import grid2op
# create an environment

from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget

from pytorch_DQN_test import DqnGrid2op

env_name = "rte_case14_realistic"  # for example, other environments might be usable
#env_name = "l2rpn_neurips_2020_track2_small"

env = grid2op.make(env_name,
                                 opponent_attack_cooldown=12*20,
                                 opponent_attack_duration=12*40,
                                 opponent_budget_per_ts=0.5,
                                 opponent_init_budget=100.,
                                 opponent_action_class=PowerlineSetAction,
                                 opponent_class=RandomLineOpponent,
                                 opponent_budget_class=BaseActionBudget,
                                 kwargs_opponent={"lines_attacked":
                                                     ['0_1_0', '0_4_1', '11_12_13', '12_13_14', '1_2_2', '1_3_3', '1_4_4', '2_3_5', '3_4_6', '3_6_15', '3_8_16', '4_5_17', '5_10_7', '5_11_8', '5_12_9', '6_7_18', '6_8_19', '8_13_11', '8_9_10', '9_10_12']}
                                 )
env.reset()

# create an agent
my_agent = DqnGrid2op(env,44)

state = env.reset()
for j in range(200):
    action = my_agent.act(state)
    env.render()
    state, reward, done, _ = env.step(action)
    print(F"state {state}, reward {reward}, done {done},")
    if done:
        break

exit()
# proceed as you would any open ai gym loop
nb_episode = 20
for _ in range(nb_episode):
    # you perform in this case 10 different episodes
    obs = env.reset()
    reward = env.reward_range[0]
    print(reward)
    done = False
    while not done:
        # here you loop on the time steps: at each step your agent receive an observation
        # takes an action
        # and the environment computes the next observation that will be used at the next step.
        _ = env.render()
        act = my_agent.act(obs, reward, done)
        print(act)
        obs, reward, done, info = env.step(act)
        print(F"{reward}")
