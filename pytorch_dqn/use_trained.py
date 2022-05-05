from grid2op.Action.PlayableAction import PlayableAction
from pytorch_dqn.Opponents import MyRandomOpponent
import torch
import grid2op
# create an environment

from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import BaseActionBudget

from agent import DqnGrid2op

env_name = "rte_case14_realistic"  # for example, other environments might be usable
# env_name = "l2rpn_neurips_2020_track2_small"

env = grid2op.make(env_name,
                   opponent_attack_cooldown=12 * 80,
                   opponent_attack_duration=12 * 10,
                   opponent_budget_per_ts=0.12,
                   opponent_init_budget=0.0,
                   opponent_action_class=PlayableAction,
                   opponent_class=MyRandomOpponent,
                   opponent_budget_class=BaseActionBudget,
                   kwargs_opponent={"lines_attacked":
                                        ['0_1_0', '0_4_1', '11_12_13', '12_13_14', '1_2_2', '1_3_3', '1_4_4', '2_3_5',
                                         '3_4_6', '3_6_15', '3_8_16', '4_5_17', '5_10_7', '5_11_8', '5_12_9', '6_7_18',
                                         '6_8_19', '8_13_11', '8_9_10', '9_10_12'],
                                    "generators_attacked":['gen_0_4', 'gen_1_0', 'gen_2_1', 'gen_5_2', 'gen_7_3']}
                   )
env.reset()

agent = DqnGrid2op(env, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('./1651696200.8540685/900_694.2073640763759_checkpoint.pth'))



for j in range(200):
    # you perform in this case 10 different episodes
    obs = env.reset()
    reward = env.reward_range[0]
    done = False
    while not done:
        # here you loop on the time steps: at each step your agent receive an observation
        # takes an action
        # and the environment computes the next observation that will be used at the next step.
        #_ = env.render()
        act = agent._act(obs, 100)
        obs, reward, done, info = env.step(agent.convert_act(act))
        print(F"act {act}, reward {reward}")

env.close()