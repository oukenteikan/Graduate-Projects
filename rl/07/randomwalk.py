#! /usr/env/python
import random
import time
import matplotlib.pyplot as plt
import pylab
import numpy as np
from sklearn.metrics import mean_squared_error
pylab.rcParams['figure.figsize'] = (15, 8)

def take_action(state, action):
    next_state = state + 2 * action - 1
    return next_state, int(next_state >= state_num)

def monte_carlo_step(states, actions):
    s = states[:]
    a = actions[:]
    next_state, reward = take_action(s[-1], a[-1])
    if next_state >= 0 and next_state < state_num:
        s.append(next_state)
        next_action = random.randint(0, 1)
        a.append(next_action)
        next_gain, next_reward = monte_carlo_step(s, a)
        gain = gamma * next_gain + next_reward
        if next_state not in states:
            R[next_state].append(gain)
            V[next_state] = V[next_state] + alpha_mc * (gain - V[next_state])
    else:
        gain = 0
    return gain, reward

def monte_carlo():
    for i in range(max_epoch):
        s0 = random.randint(0, state_num-1)
        a0 = random.randint(0, 1)
        next_gain, reward = monte_carlo_step([s0], [a0])
        gain = gamma * next_gain + reward
        R[s0].append(gain)
        V[s0] = V[s0] + alpha_mc * (gain - V[s0])
        E[i] = np.linalg.norm(np.array(V)-X, ord=2)
        E[i] = mean_squared_error(V, X)
    #for state in range(state_num): print(R[state])
    print(V)
    return

def temporal_difference_step(states):
    state = states[-1]
    s = states[:]
    action = random.randint(0, 1)
    next_state, reward = take_action(state, action)
    if next_state >= 0 and next_state < state_num:
        gain = gamma * V[next_state] + reward
        R[state].append(gain)
        V[state] = V[state] + alpha_td * (gain - V[state])
        s.append(next_state)
        temporal_difference_step(s)
    else:
        gain = reward
        R[state].append(gain)
        V[state] = V[state] + alpha_td * (gain - V[state])
    return

def temporal_difference():
    for i in range(max_epoch):
        s0 = random.randint(0, state_num-1) 
        temporal_difference_step([s0])
        E[i] = np.linalg.norm(np.array(V)-X, ord=2)
        E[i] = mean_squared_error(V, X)
    #for state in range(state_num): print(R[state])
    print(V)
    return

def dynamic():
    b = np.array([0, 0, 0, 0, 0.5])
    A = np.eye(5) - 0.5*gamma*np.eye(5, k=1) - 0.5*gamma*np.eye(5, k=-1)
    X = np.linalg.solve(A, b)
    print(type(X))
    print(X)
    return X

if __name__ == "__main__":
    state_num = 5
    max_epoch = 100
    gamma = 1
    X = dynamic()
    E = [0] * max_epoch

    alpha_mc = 0.01
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    monte_carlo()
    plt.plot(range(max_epoch), E, label = "monte_carlo: alpha = {}".format(alpha_mc))

    alpha_mc = 0.02
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    monte_carlo()
    plt.plot(range(max_epoch), E, label = "monte_carlo: alpha = {}".format(alpha_mc))

    alpha_mc = 0.05
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    monte_carlo()
    plt.plot(range(max_epoch), E, label = "monte_carlo: alpha = {}".format(alpha_mc))

    alpha_mc = 0.10
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    monte_carlo()
    plt.plot(range(max_epoch), E, label = "monte_carlo: alpha = {}".format(alpha_mc))

    alpha_mc = 0.16
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    monte_carlo()
    plt.plot(range(max_epoch), E, label = "monte_carlo: alpha = {}".format(alpha_mc))

    alpha_td = 0.01
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    temporal_difference()
    plt.plot(range(max_epoch), E, label = "temporal_difference: alpha = {}".format(alpha_td))

    alpha_td = 0.02
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    temporal_difference()
    plt.plot(range(max_epoch), E, label = "temporal_difference: alpha = {}".format(alpha_td))

    alpha_td = 0.05
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    temporal_difference()
    plt.plot(range(max_epoch), E, label = "temporal_difference: alpha = {}".format(alpha_td))

    alpha_td = 0.10
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    temporal_difference()
    plt.plot(range(max_epoch), E, label = "temporal_difference: alpha = {}".format(alpha_td))

    alpha_td = 0.16
    R = [[] for i in range(state_num)]
    V = [0] * state_num
    temporal_difference()
    plt.plot(range(max_epoch), E, label = "temporal_difference: alpha = {}".format(alpha_td))

    plt.legend()
    plt.savefig("randomwalk.png")
    plt.show()
