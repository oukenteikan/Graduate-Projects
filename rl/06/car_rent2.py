import math
import copy
import numpy as np

car_number = 20
max_trans = 5
max_poisson = 15
discount_rate = 0.9
lambda_rent1 = 3
lambda_rent2 = 4
lambda_return1 = 3
lambda_return2 = 2

print("Notice that the poisson distribution has a very small possibility when X > 15 with lambda 2, 3 or 4, so we set the max value as {}, then do a normalization.".format(max_poisson))
possibility = np.array([[[[math.pow(lambda_rent1, rent1) * math.pow(lambda_rent2, rent2)
               * math.pow(lambda_return1, return1) * math.pow(lambda_return2, return2)
               / math.factorial(rent1) / math.factorial(rent2)
               / math.factorial(return1) / math.factorial(return2)
               * math.exp(- lambda_rent1 - lambda_rent2 - lambda_return1 - lambda_return2)
                  for return2 in range(max_poisson)]
                 for return1 in range(max_poisson)]
                for rent2 in range(max_poisson)]
               for rent1 in range(max_poisson)])
possibility /= np.sum(possibility)
print(possibility)

threshold = np.zeros((car_number+1, car_number+1)) + 0.1
v = np.zeros((car_number+1, car_number+1))
a = np.zeros((car_number+1, car_number+1), dtype=np.int)

while True:
    while True:
        new_v = np.zeros((car_number+1, car_number+1))
        for i in range(car_number+1):
            for j in range(car_number+1):
                print("i = {} j = {}".format(i, j))
                left1 = i - a[i][j]
                left2 = j + a[i][j]
                for rent1 in range(max_poisson):
                    real_rent1 = min(rent1, left1)
                    for rent2 in range(max_poisson):
                        real_rent2 = min(rent2, left2)
                        reward = 10 * (real_rent1+real_rent2) - 2 * abs(a[i][j])
                        for return1 in range(max_poisson):
                            real_return1 = min(return1, car_number - (left1 - real_rent1))
                            for return2 in range(max_poisson):
                                real_return2 = min(return2, car_number - (left2 - real_rent2))
                                real_left1 = left1 - real_rent1 + real_return1
                                real_left2 = left2 - real_rent2 + real_return2
                                new_v[i][j] += possibility[rent1][rent2][return1][return2] * (reward + discount_rate * v[real_left1][real_left2])
        print("old v = {}".format(v))
        print("new v = {}".format(new_v))
        if (np.abs(new_v - v) < threshold).all():
            break
        v = new_v
    new_a = copy.deepcopy(a)
    for i in range(car_number+1):
        for j in range(car_number+1):
            for action in range(-min(j, max_trans), min(i, max_trans)+1):
                print("i = {} j = {} action = {}".format(i, j, action))
                new_value = 0
                left1 = i - action
                left2 = j + action
                for rent1 in range(max_poisson):
                    real_rent1 = min(rent1, left1)
                    for rent2 in range(max_poisson):
                        real_rent2 = min(rent2, left2)
                        reward = 10 * (real_rent1+real_rent2) - 2 * abs(action)
                        for return1 in range(max_poisson):
                            real_return1 = min(return1, car_number - (left1 - real_rent1))
                            for return2 in range(max_poisson):
                                real_return2 = min(return2, car_number - (left2 - real_rent2))
                                real_left1 = left1 - real_rent1 + real_return1
                                real_left2 = left2 - real_rent2 + real_return2
                                new_value += possibility[rent1][rent2][return1][return2] * (reward + discount_rate * v[real_left1][real_left2])
                if new_value > new_v[i][j]:
                    new_v[i][j] = new_value
                    new_a[i][j] = action
    print("old a = {}".format(a))
    print("new a = {}".format(new_a))
    if (new_a == a).all():
        break
    a = new_a

print("final v = {}".format(new_v))
print("final a = {}".format(new_a))
print("Final v and a is shown above which mean when there are i and j cars at location 1 and 2 at the end of the day, you can get at most v[i][j] benefit and you should move a[i][j] from location 1 to 2.")
