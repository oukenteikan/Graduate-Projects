#! /usr/env/python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
pylab.rcParams['figure.figsize'] = (15, 8)

def print3d(valuefunction):
    global figureIndex
    gridsize = 40
    positionstep = (max_position - min_position)/gridsize
    positions = np.arange(min_position, max_position + positionstep, positionstep)
    velocitystep = (max_velocity - min_velocity)/gridsize
    velocities = np.arange(min_velocity, max_velocity + velocitystep, velocitystep)
    x = []
    y = []
    z = []
    for i in positions:
        for j in velocities:
            x.append(i)
            y.append(j)
            z.append(valuefunction.get_cost(i, j))
    figureIndex += 1
    title = 'Episode:'+str(index)
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_zlabel('cost to go')
    plt.savefig(str(figureIndex)+".png")
    plt.show()


f = open("log2.log", 'r')
lines = f.readlines()
start = 0
while lines[start].find("final v") == -1: start += 1
print(start)
print(lines[start])
end = start
while lines[end].find("final a") == -1: end += 1
print(end)
print(lines[end])
v = lines[start:end]
length = end - start
l = v[0].find('[')
v[0] = v[0][l:]
for i in range(length):
    v[i] = v[i].strip().strip('[]').split()
V = [v[i][j] for i in range(length) for j in range(len(v[i]))]
print(len(V))
print(V)
V = np.array(V, dtype='float')
V = V.reshape((21, 21))
print(V.shape)
print(V)
x = []
y = []
z = []
for i in range(21):
    for j in range(21):
        x.append(i)
        y.append(j)
        z.append(V[i][j])
fig = plt.figure()
title = 'Car Rent'
fig.suptitle(title)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('location 1')
ax.set_ylabel('location 2')
ax.set_zlabel('reward')
plt.savefig("car_rent.png")
plt.show()

