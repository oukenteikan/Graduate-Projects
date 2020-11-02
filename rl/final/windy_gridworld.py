#！ /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import inspect
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pylab
import prettytable
pylab.rcParams['figure.figsize'] = (15, 8)

logging.basicConfig(filename = 'log.txt', 
                    filemode = 'a', 
                    format=f'%(asctime)s %(filename)s %(funcName)s [%(lineno)d]: %(message)s', 
                    datefmt="%Y-%M-%d %H:%M:%S", 
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.warning('Setting logger done.')

row_num = 7
strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
col_num = len(strength)
vary = [1, 0, -1]
var_num = len(vary)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
act_num = len(actions)
king_actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
king_act_num = len(king_actions)
start_point = (3, 0)
goal_point = (3, 7)
logger.warning('Setting environment done.')

ALPHA = 0.5
GAMMA = 0.9
EPSILON = 0.1
EPISODE = 1000
STOCHASTIC = False
KING = True
MAX_STEP = 100
ROUND = 2
UNVISITED = 'Never reach'
FAILURE = 'Fail to reach'
action_names = ['up', 'down', 'left', 'right']
king_action_names = ['up', 'down', 'left', 'right', 'upleft', 'downleft', 'upright', 'downright']
fancy_action_names = ['↑', '↓', '←', '→']
fancy_king_action_names = ['↑', '↓', '←', '→', '↖', '↙', '↗', '↘']
logger.warning('Setting hyperparameters done.')

def choose_actions(Q, S):
    candidates = Q[S[0]][S[1]]
    max_candidates = np.where(candidates == np.max(candidates))
    return max_candidates[0]

def choose_action(Q, S, epsilon = 0, king = KING):
    candidates = choose_actions(Q, S)
    A = candidates[random.randint(0, candidates.shape[0]-1)]
    if epsilon and random.random() < epsilon:
        A = random.randint(0, king_act_num-1) if king else random.randint(0, act_num-1)
    return A

def take_action(Q, S, A, visit, stochastic = STOCHASTIC, king = KING):
    action = king_actions[A] if king else actions[A]
    x = S[0] + action[0] - strength[S[1]]
    y = S[1] + action[1]
    offset = vary[random.randint(0, var_num-1)] if stochastic and strength[S[1]] else 0
    x = max(min(x - offset, row_num-1), 0)
    y = max(min(y, col_num-1), 0)
    visit[x][y] = 1
    S_ = (x, y)
    R = 0 if S_ == goal_point else -1
    return R, S_

def update_value(Q, S, A, R, S_, A_, alpha = ALPHA, gamma = GAMMA):
    Q[S[0]][S[1]][A] = (1 - alpha) * Q[S[0]][S[1]][A] + alpha * (R + gamma * Q[S_[0]][S_[1]][A_])
    return

def draw_figure(episode, steps, title):
    logger.debug(steps)
    #steps = np.cumsum(steps)
    plt.plot(range(episode), steps, label = title)
    plt.ylim(ymin=0, ymax=15)
    plt.legend()
    filename = f"figures/windy_gridworld_{title}_{episode}.png"
    plt.savefig(filename)
    logger.warning(f'Saving plot {filename} done')
    plt.show()
    return

def draw_table(Q, visit, title, king = KING):

    strategy_table = prettytable.PrettyTable()
    fancy_table = prettytable.PrettyTable()
    reward_table = prettytable.PrettyTable()
    strategy_table.title = title
    fancy_table.title = title
    reward_table.title = title 
    logger.debug(title)

    head = list(range(col_num))
    head.insert(0, 'row/col')
    logger.debug(head)
    strategy_table.field_names = head
    fancy_table.field_names = head
    reward_table.field_names = head

    for i in range(row_num):
        strategy_row = [i]
        reward_row = [i]
        fancy_row = [i]
        for j in range(col_num):
            candidates = choose_actions(Q, (i, j))
            if king:
                candidate_names = '_'.join([king_action_names[A] for A in candidates]) if visit[i][j] else UNVISITED
                fancy_candidate_names = ' '.join([fancy_king_action_names[A] for A in candidates]) if visit[i][j] else UNVISITED
            else:
                candidate_names = '_'.join([action_names[A] for A in candidates]) if visit[i][j] else UNVISITED
                fancy_candidate_names = ' '.join([fancy_action_names[A] for A in candidates]) if visit[i][j] else UNVISITED
            strategy_row.append(candidate_names)
            fancy_row.append(fancy_candidate_names)
            reward_row.append(round(Q[i][j][candidates[0]], ndigits=ROUND))
        logger.debug(strategy_row)
        logger.debug(fancy_row)
        logger.debug(reward_row)
        strategy_table.add_row(strategy_row)
        reward_table.add_row(reward_row)
        fancy_table.add_row(fancy_row)

    logger.warning(f'Optimal strategy table: \n{strategy_table}')
    logger.warning(f'Optimal visualized table: \n{fancy_table}')
    logger.warning(f'Optimal reward table: \n{reward_table}')

    filename = f"tables/windy_gridworld_{title}.txt"
    with open(filename, 'w') as f:
        f.write(str(strategy_table) + '\n\n\n' + str(fancy_table) + '\n\n\n' + str(reward_table))
    return

def evaluate(Q, visit, gamma = GAMMA, stochastic = STOCHASTIC, king = KING):
    total_reward = 0
    S = start_point
    step = 0
    logger.debug(f'Evaluate: Start from point {S}')
    while S != goal_point:
        A = choose_action(Q, S, 0, king)
        R, S_ = take_action(Q, S, A, visit, stochastic, king)
        logger.debug(f'Evaluate: Take action {A} from point {S} and get reward {R} ending with point {S_}')
        total_reward += (gamma ** step) * R
        S = S_
        step += 1
        if step > MAX_STEP:
            total_reward = FAILURE
            break
    logger.debug(f'Evaluate: Arrive at point {S} after {step} steps')
    logger.debug(f'Evaluate: Total reward is {total_reward}')
    return

def q_learning(alpha = ALPHA, gamma = GAMMA, epsilon = EPSILON, episode = EPISODE, stochastic = STOCHASTIC, king = KING):

    method = inspect.stack()[0][3]
    logger.warning(f'Method       : {method}')
    logger.warning(f'Stochastic   : {stochastic}')
    logger.warning(f'King\'s move  : {king}')
    logger.warning(f'Configuration: alpha = {alpha} gamma = {gamma} epsilon = {epsilon} episode = {episode}')

    if king:
        Q = np.zeros(shape = (row_num, col_num, king_act_num), dtype = float)
    else:
        Q = np.zeros(shape = (row_num, col_num, act_num), dtype = float)

    visit = np.zeros(shape = (row_num, col_num), dtype = float)
    visit[start_point[0]][start_point[1]] = 1
    steps = np.zeros(shape = episode, dtype = int)

    for e in range(episode):
        S = start_point
        logger.debug(f'Episode {e}: Start from point {S}')
        while S != goal_point:
            steps[e] += 1
            A = choose_action(Q, S, epsilon, king)
            R, S_ = take_action(Q, S, A, visit, stochastic, king)
            logger.debug(f'Episode {e}: Take action {A} from point {S} and get reward {R} ending with point {S_}')
            A_ = choose_action(Q, S_, 0, king)
            update_value(Q, S, A, R, S_, A_, alpha, gamma)
            S = S_            
        logger.debug(f'Episode {e}: Arrive at point {S} after {steps[e]} steps')
    evaluate(Q, visit, gamma, stochastic, king)

    title = f"{method}_stochastic_{stochastic}_king_{king}"
    draw_figure(episode, steps, title)
    draw_table(Q, visit, title, king)
    return

def sarsa(alpha = ALPHA, gamma = GAMMA, epsilon = EPSILON, episode = EPISODE, stochastic = STOCHASTIC, king = KING):

    method = inspect.stack()[0][3]
    logger.warning(f'Method       : {method}')
    logger.warning(f'Stochastic   : {stochastic}')
    logger.warning(f'King\'s move : {king}')
    logger.warning(f'Configuration: alpha = {alpha} gamma = {gamma} epsilon = {epsilon} episode = {episode}')

    if king:
        Q = np.zeros(shape = (row_num, col_num, king_act_num), dtype = float)
    else:
        Q = np.zeros(shape = (row_num, col_num, act_num), dtype = float)

    visit = np.zeros(shape = (row_num, col_num), dtype = float)
    visit[start_point[0]][start_point[1]] = 1
    steps = np.zeros(shape = episode, dtype = int)

    for e in range(episode):
        S = start_point
        logger.debug(f'Episode {e}: Start from point {S}')
        while S != goal_point:
            A = choose_action(Q, S, epsilon, king)
            R, S_ = take_action(Q, S, A, visit, stochastic, king)
            logger.debug(f'Episode {e}: Take action {A} from point {S} and get reward {R} ending with point {S_}')
            A_ = choose_action(Q, S_, epsilon, king)
            update_value(Q, S, A, R, S_, A_, alpha, gamma)
            S = S_
            steps[e] += 1
        logger.debug(f'Episode {e}: Arrive at point {S} after {steps[e]} steps')
    evaluate(Q, visit, gamma, stochastic, king)

    title = f"{method}_stochastic_{stochastic}_king_{king}"
    draw_figure(episode, steps, title)
    draw_table(Q, visit, title, king)
    return

def main():
    q_learning()
    sarsa()
    return

if __name__ == '__main__':
    main()