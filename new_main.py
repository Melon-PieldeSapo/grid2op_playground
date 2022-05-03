import os
import time

from RandomLineOpponent import MyRandomLineOpponent
from pytorch_DQN_test import DqnGrid2op
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import grid2op
# create an environment

from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget

from pytorch_DQN_test import DqnGrid2op

env_name = "rte_case14_realistic"  # for example, other environments might be usable
runtime = time.time()
# env_name = "l2rpn_neurips_2020_track2_small"
os.mkdir(F"./{runtime}")

env = grid2op.make(env_name,
                   opponent_attack_cooldown=12 * 40,
                   opponent_attack_duration=12 * 5,
                   opponent_budget_per_ts=0.5,
                   opponent_init_budget=0.0,
                   opponent_action_class=PowerlineSetAction,
                   opponent_class=MyRandomLineOpponent,
                   opponent_budget_class=BaseActionBudget,
                   kwargs_opponent={"lines_attacked":
                                        ['0_1_0', '0_4_1', '11_12_13', '12_13_14', '1_2_2', '1_3_3', '1_4_4', '2_3_5',
                                         '3_4_6', '3_6_15', '3_8_16', '4_5_17', '5_10_7', '5_11_8', '5_12_9', '6_7_18',
                                         '6_8_19', '8_13_11', '8_9_10', '9_10_12']}
                   )

env.reset()

agent = DqnGrid2op(env, seed=0)


def dqn(n_episodes=4000, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):

            # elegir accion At con politica e-greedy
            action = agent.act(state, eps)

            # aplicar At y obtener Rt+1, St+1
            next_state, reward, done, _ = env.step(agent.convert_act(action))

            # almacenar <St, At, Rt+1, St+1>
            agent.memory.add(agent.convert_obs(state), action, reward, agent.convert_obs(next_state), done)

            # train & update
            agent.step(agent.convert_obs(state), action, reward, agent.convert_obs(next_state), done)

            # avanzar estado
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)  # guardar ultima puntuacion
        scores.append(score)  # guardar ultima puntuacion
        eps = max(eps_end, eps_decay * eps)  # reducir epsilon

        print('\rEpisodio {}\tPuntuacion media (ultimos {:d}): {:.2f}'.format(i_episode, 100, np.mean(scores_window)),
              end="")
        if i_episode % 100 == 0:
            print('\rEpisodio {}\tPuntuacion media ({:d} anteriores): {:.2f}'.format(i_episode, 100,
                                                                                     np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(),
                       F"{runtime}/{i_episode}_{np.mean(scores_window)}_checkpoint.pth")  # guardar pesos de agente entrenado

        if np.mean(scores_window) >= 300000.0:
            print('\nProblema resuelto en {:d} episodios!\tPuntuacion media (ultimos {:d}): {:.2f}'.format(
                i_episode - 100, 100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')  # guardar pesos de agente entrenado
            break
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
    while not done:
        # here you loop on the time steps: at each step your agent receive an observation
        # takes an action
        # and the environment computes the next observation that will be used at the next step.
        _ = env.render()
        act = agent.act(obs, j)
        print(act)
        obs, reward, done, info = env.step(agent.convert_act(act))
        print(F"{reward}")
