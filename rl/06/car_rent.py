import math
import copy
import numpy as np

car_number = 20
max_trans = 5
max_poisson = 10
lambda_rent1 = 3
lambda_rent2 = 4
lambda_return1 = 3
lambda_return2 = 2

threshold = np.zeros((car_number+1, car_number+1)) + 0.1
print(threshold)

possibility = np.array([[[[math.pow(lambda_rent1, rent1) * math.pow(lambda_rent2, rent2)
               * math.pow(lambda_return1, return1) * math.pow(lambda_return2, return2)
               / math.factorial(rent1) / math.factorial(rent2)
               / math.factorial(return1) / math.factorial(return2)
               * math.exp(- lambda_rent1 - lambda_rent2 - lambda_return1 - lambda_return2)
                  for return2 in range(max_poisson)]
                 for return1 in range(max_poisson)]
                for rent2 in range(max_poisson)]
               for rent1 in range(max_poisson)])
print(possibility)

v = np.zeros((car_number+1, car_number+1))
a = np.zeros((car_number+1, car_number+1), dtype=np.int)

while True:
    while True:
        new_v = np.zeros((car_number+1, car_number+1))
        for i in range(car_number+1):
            for j in range(car_number+1):
                print("i ={} j ={}".format(i, j))
                new_v[i][j] = np.sum([[[[possibility[rent1][rent2][return1][return2]
                                      * (10 * (min(rent1, i-a[i][j])+min(rent2, j+a[i][j])) - 2 * abs(a[i][j])
                                         + 0.9 * v[i - a[i][j] - min(rent1, i-a[i][j]) + min(return1, car_number-(i-a[i][j]-min(rent1, i-a[i][j])))]
                                         [j + a[i][j] - min(rent2, j+a[i][j]) + min(return2, car_number-(j+a[i][j]-min(rent2, j+a[i][j])))])
                                         for return2 in range(max_poisson)]
                                        for return1 in range(max_poisson)]
                                       for rent2 in range(max_poisson)]
                                      for rent1 in range(max_poisson)])
        print("v = {}".format(new_v))
        if (np.abs(new_v - v) < threshold).all():
            break
        v = new_v
    new_a = copy.deepcopy(a)
    for i in range(car_number+1):
        for j in range(car_number+1):
            for action in range(-min(j, max_trans), min(i, max_trans)+1):
                print("i ={} j ={} action ={}".format(i, j, action))
                new_value = np.sum([[[[possibility[rent1][rent2][return1][return2]
                                      * (10 * (min(rent1, i-action)+min(rent2, j+action)) - 2 * abs(action)
                                         + 0.9 * v[i - action - min(rent1, i-action) + min(return1, car_number-(i-action-min(rent1, i-action)))]
                                         [j + action - min(rent2, j+action) + min(return2, car_number-(j+action-min(rent2, j+action)))])
                                         for return2 in range(max_poisson)]
                                        for return1 in range(max_poisson)]
                                       for rent2 in range(max_poisson)]
                                      for rent1 in range(max_poisson)])
                if new_value > new_v[i][j]:
                    new_v[i][j] = new_value
                    new_a[i][j] = action
    print("a = {}".format(new_a))
    if (new_a == a).all():
        break
    a = new_a

print("final v = {}".format(new_v))
print("final a = {}".format(new_a))

