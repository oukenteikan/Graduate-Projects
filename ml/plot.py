#! /usr/bin/env/ python
# coding=utf8

import matplotlib as  mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

x = np.arange(0, 1, 0.001)
t1 = 0.5
t2 = 0.95

def naive_iou():
    y = x > t1
    plt.plot(x, y, label = '简单IoU规则')

def nms_iou():
    y = x >= t2
    plt.plot(x, y, label = 'IoU规则加入NMS机制')

def nsquare():
    y = np.clip((x-t1) / (t2-t1), 0, 1)
    y = 1 - (1-y) ** 3
    plt.plot(x, y, label = '负三次函数')

def linear():
    y = np.clip((x-t1) / (t2-t1), 0, 1)
    plt.plot(x, y, label = '线性函数')

def tripple():
    y = np.clip((x-t1) / (t2-t1), 0, 1)
    y = ((2*y-1)**3 + 1) * 0.5
    plt.plot(x, y, label = '三次函数')

def square():
    y = np.clip((x-t1) / (t2-t1), 0, 1)
    y = y * y
    plt.plot(x, y, label = '二次函数')

def triangle():
    y = np.clip((x-t1) / (t2-t1), 0, 1)
    y = np.sin(y * math.pi / 2)
    plt.plot(x, y, label = 'sin函数')

def sigmoid(alpha=8):
    y = (x-t1) / (t2-t1)
    temp = torch.tensor(y)
    temp = torch.sigmoid(alpha * (temp-0.5))
    temp = temp.clamp(min=0, max=1)
    y = temp.data.numpy()
    plt.plot(x, y, label = 'sigmoid函数')

if __name__ == '__main__':
    plt.title("概率函数:t1=0.5 t2=0.95")
    naive_iou()
    nms_iou()
    linear()
    square()
    #nsquare()
    triangle()
    sigmoid(16)
    tripple()
    plt.legend()
    plt.savefig('./figure.png')
    plt.show()
    
    
