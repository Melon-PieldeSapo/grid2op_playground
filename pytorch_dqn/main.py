import os
import random
import time

from pytorch_dqn.Opponents import MyRandomOpponent
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import grid2op
# create an environment

from grid2op.Action import PowerlineSetAction, PlayableAction
from grid2op.Opponent import BaseActionBudget

from agent import DqnGrid2op

env_name = "rte_case14_realistic"  # for example, other environments might be usable
runtime = time.time()
# env_name = "l2rpn_neurips_2020_track2_small"
os.mkdir(F"./{runtime}")

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
                                    "generators_attacked": ['gen_0_4', 'gen_1_0', 'gen_2_1', 'gen_5_2', 'gen_7_3']}
                   )
env.reset()

agent = DqnGrid2op(env, seed=0)


def dqn(n_episodes=2000, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): numero maximo de episodios de entrenamiento (n_episodios)
        max_t (int): numero maximo de pasos por episodio (n_entrenamiento)
        eps_start (float): valor inicial de epsilon
        eps_end (float): valor final de epsilon
        eps_decay (float): factor de multiplicacion (por episodio) de epsilon
    """
    scores = []  # puntuaciones de cada episodio
    scores_window = deque(maxlen=100)  # puntuaciones de los ultimos 100 episodios
    eps = eps_start  # inicializar epsilon
    t_sum = 0
    actions = {}
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        actions[i_episode] = []
        for t in range(max_t):

            # elegir accion At con politica e-greedy
            action = agent._act(state, eps)

            # aplicar At y obtener Rt+1, St+1
            next_state, reward, done, _ = env.step(agent.convert_act(action))
            reward = agent.norm_reward(reward)
            actions[i_episode].append((action, reward))
            # almacenar <St, At, Rt+1, St+1>
            agent.memory.add(agent.convert_obs(state), action, reward, agent.convert_obs(next_state), done)
            if action != 0 or random.random() > 0.1:
                # train & update
                agent.step(agent.convert_obs(state), action, reward, agent.convert_obs(next_state), done)

            # avanzar estado
            state = next_state
            score += reward

            if done:
                break
        t_sum += t
        scores_window.append(score)  # guardar ultima puntuacion
        scores.append(score)  # guardar ultima puntuacion
        eps = max(eps_end, eps_decay * eps)  # reducir epsilon

        print(
            F'\rEpisodio {i_episode}\tPuntuacion media (ultimos {100:d}): {np.mean(scores_window):.2f}\t Done in {t} t.'
            F' Actions: {actions[i_episode]}',
            end="")
        if i_episode % 100 == 0:
            print(
                F'\rEpisodio {i_episode}\tPuntuacion media ({100:d} anteriores): {np.mean(scores_window):.2f}\t Done in {t_sum / 100} mean T.'
                F' Actions: {actions[i_episode]}')
            t_sum = 0
            torch.save(agent.qnetwork_local.state_dict(),
                       F"{runtime}/{i_episode}_{np.mean(scores_window)}_checkpoint.pth")  # guardar pesos de agente entrenado

        if np.mean(scores_window) >= 2000.0:
            print(F'\nProblema resuelto en {i_episode - 100:d} episodios!\tPuntuacion media (ultimos {100:d}): { np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')  # guardar pesos de agente entrenado
            break
    print(actions)
    return scores


scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Puntuacion')
plt.xlabel('Episodio #')
plt.show()

for j in range(200):
    # you perform in this case 10 different episodes
    obs = env.reset()
    reward = env.reward_range[0]
    print(reward)
    done = False
    score = 0
    while not done:
        # here you loop on the time steps: at each step your agent receive an observation
        # takes an action
        # and the environment computes the next observation that will be used at the next step.
        _ = env.render()
        act = agent._act(obs, 100)
        print(F"act: {act}")
        obs, reward, done, info = env.step(agent.convert_act(act))
        print(F"gives reward: {reward}")
        score += reward
    print(F"J: {j} gives reward: {score}")
