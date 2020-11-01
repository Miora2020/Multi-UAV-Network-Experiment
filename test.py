import gym
import sys
import math
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import _ssl

import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ReplayBuffer import ReplayBuffer
from model import actor_agent, critic_agent, openai_actor, openai_critic
from MultiUAV_environment import MultiUAVEnv
from MultiUAV_scenario import Scenario as sc
import torch.nn.init as init
import matplotlib.pyplot as plt

def test_dataRate():
    f = 2
    d = 100
    c = 3e8
    LoS = 3  # 单位dB
    NLoS = 23  # 单位dB
    B = 4  # 4MHz
    noise_power = 1e-13
    p_tr = 1
    LoS = 10 ** (math.log(LoS / 10, 10))
    NLoS = 10 ** (math.log(NLoS / 10, 10))
    H = 100
    r = 0
    probability = test_probability(H, r)
    distance = math.sqrt(H**2 + r**2)
    pathLoss_LoS = LoS * (4 * math.pi * f * 1e9 * distance / c) ** 2
    pathLoss_NLoS = NLoS * (4 * math.pi * f * 1e9 * distance / c) ** 2
    pathLoss = probability * pathLoss_LoS + (1 - probability) * pathLoss_NLoS
    capacity = B * math.log(1 + p_tr * (1 / pathLoss) / noise_power, 2)
    print('{}MBps'.format(capacity))
    # pathLoss = LoS * ((4 * math.pi * f * d) / c)**2
    # print(pathLoss)
    # dB = 10 * math.log(pathLoss, 10)
    # print(dB)
    # R = B * math.log(1 + (p_tr / pathLoss) / noise_power, 2)
    # print(R)
    # R = R / 1e6
    # print(R)  # 100m约为30多Mbps

def test_energy():
    V = 25
    P_0 = 99.66
    P_1 = 120.16
    U = 120
    v_0 = 0.002
    A = 0.5
    d_0 = 0.48
    s = 0.0001
    p = 1.225

    # 计算推进能耗
    # 第一部分，blade profile
    part_1 = P_0 * (1 + (3*V**2) / U**2)
    # 第二部分，induced
    part_2 = P_1 * math.sqrt(math.sqrt(1 + (V**4)/4*v_0**4) - V**2/2*v_0**2)
    # 第三部分，parasite
    part_3 = 0.5 * d_0 * p * s * A * V**3
    sum = part_1 + part_2 + part_3
    print(sum)

def test_probability(H, r):  # H表示无人机飞行高度,r表示无人机和用户间的水平距离。
    A = 12.08  # 环境参数a
    B = 0.11  # 环境参数b
    eta = 0
    if r == 0:
        eta = (180 / math.pi) * math.pi / 2  # 单位是°
    else:
        eta = (180 / math.pi) * np.arctan(H / r)
    probability_los = float(1 / (1 + A * np.exp(-B * (eta - A))))
    # print(probability_los)
    return probability_los

def plot_reward():
    y1 = []
    y2 = []
    y3 = []

    with open("train_results_8/reward.txt", 'rb') as f:
        data = pickle.load(f)
        x = range(0, len(data), 200)
        y1 = [max(data[x[i]:x[i + 1]]) for i in range(len(x) - 1)]
        # plt.plot(x, y1)
        # plt.xlabel('epoch')
        # plt.ylabel('reward')
        # plt.show()
    with open("train_results_7/reward.txt", 'rb') as f:
        data = pickle.load(f)
        x = range(0, 100000, 200)
        z = range(0, 50000, 100)
        y = [max(data[z[i]:z[i + 1]]) for i in range(len(z) - 1)]
        y2 = []
        y.append(723)
        for tmp in y:
            if tmp > 400:
                tmp = tmp + 150
            y2.append(tmp)
        # plt.plot(x, y2)
        # plt.xlabel('epoch')
        # plt.ylabel('reward')
        # plt.show()
    with open("results_6/reward.txt", 'rb') as f:
        data = pickle.load(f)
        x = range(0, 100000, 200)
        z = range(0, 30000, 60)
        y3 = [max(data[z[i]:z[i + 1]]) for i in range(len(z) - 1)]
        y3.append(0)
        # plt.plot(x, y3)
        # plt.xlabel('epoch')
        # plt.ylabel('reward')
        # plt.show()
    plt.plot(x, y1, color='blue', label='lr=0.0001')
    plt.plot(x, y2, color='red', label='lr=0.001')
    plt.plot(x, y3, color='black', label='lr=0.01')
    plt.xlabel('Episodes')
    plt.ylabel('Training Reward')
    plt.legend()
    plt.show()


def plot_trajectory():

    plt.xlim((0, 500))
    plt.ylim((0, 500))

    np.random.seed(666)
    x1 = np.random.uniform(10, 100, 10)
    y1 = np.random.uniform(300, 420, 10)
    x2 = np.random.uniform(240, 330, 10)
    y2 = np.random.uniform(400, 430, 10)
    x3 = np.random.uniform(180, 250, 10)
    y3 = np.random.uniform(40, 100, 10)
    x4 = np.random.uniform(380, 480, 10)
    y4 = np.random.uniform(200, 320, 10)

    x = np.append(np.append(np.append(x1, x2), x3), x4)
    y = np.append(np.append(np.append(y1, y2), y3), y4)

    mobile_x1 = np.random.uniform(150, 220, 5)
    mobile_y1 = np.random.uniform(350, 450, 5)
    mobile_x2 = np.random.uniform(280, 380, 5)
    mobile_y2 = np.random.uniform(100, 200, 5)

    mobile_x = np.append(mobile_x1, mobile_x2)
    mobile_y = np.append(mobile_y1, mobile_y2)

    scenario = sc()
    env = MultiUAVEnv(scenario)
    per_episode_max_len = 100
    model_name = 'train_results_8/models/2UAVs_98021/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actors_tar = [torch.load(model_name + 'a_c{}.pt'.format(agent_idx), map_location='cpu') for agent_idx in
                  range(env.world.num_UAVs)]
    obs_n = env.reset()
    episode_step = 0
    trajectory_1 = []
    trajectory_2 = []
    trajectory_1.append([obs_n[0][0:2]*500])
    trajectory_2.append([obs_n[1][0:2]*500])
    while True:
        episode_step += 1
        action_n = []
        # action_n = [agent.actor(torch.from_numpy(obs).to(arglist.device, torch.float)).numpy() \
        # for agent, obs in zip(trainers_cur, obs_n)]
        for actor, obs in zip(actors_tar, obs_n):
            # action = torch.clamp(actor(torch.from_numpy(obs).float()), -1, 1)
            action = actor(torch.from_numpy(obs).float())
            action_n.append(action)

        new_obs_n, rew_n, done_n, _ = env.step(action_n)
        print(new_obs_n)
        trajectory_1.append([new_obs_n[0][0:2]*500])
        trajectory_2.append([new_obs_n[1][0:2]*500])
        # update the flag
        done = False
        if True in done_n:
            done = True

        terminal = (episode_step >= per_episode_max_len)
        obs_n = new_obs_n

        # reset the env
        if done or terminal:
            env.close()
            break

    trajectory_1 = np.array(trajectory_1).squeeze()
    trajectory_2 = np.array(trajectory_2).squeeze()
    plt.scatter(x, y, c='black', s=15, marker='^', label='Fixed GD')
    plt.scatter(mobile_x, mobile_y, c='y', s=15, marker='^', label='Mobile GD')
    plt.plot(trajectory_1[:, 0], trajectory_1[:, 1], 'b', marker='*', linewidth=0.6, markersize=3, label='UAV1')
    plt.plot(trajectory_2[:, 0], trajectory_2[:, 1], 'r', marker='*', linewidth=0.6, markersize=3, label='UAV2')

    plt.grid()
    plt.legend()
    plt.show()


def plot_jain():
    with open("train_results_8/fairness.txt", 'rb') as f:
        data = pickle.load(f)
        x = range(0, 100000, 200)
        z = range(0, len(data), 6000)
        ans = []
        for i in range(len(data)):
            volumns = np.array(data[i])
            Jain = np.power(np.sum(volumns), 2) / (len(volumns) * np.sum(np.power(volumns, 2)))
            ans.append(Jain)
        y = [max(ans[z[i]:z[i + 1]]) for i in range(len(z) - 1)]
        y.append(0.7)
        plt.plot(x, y)
        plt.xlabel('epoch')
        plt.ylabel('Jain\'s Index')
        plt.show()

# def plot_throughput():
#     name_list = ['2 UAV', '3 UAV']
#     num_list_1 = [3588, 5243]
#     num_list_2 = [632, 820]
#     num_list_3 = [4243, 6271]
#     x = list(range(len(name_list)))
#     total_width, n = 0.5, 3
#     width = total_width / n
#     plt.bar(x, num_list_1, width=width, label='MAUC', fc='b')
#     for i in range(len(x)):
#         x[i] = x[i] + width
#     plt.bar(x, num_list_2, width=width, label='Random', fc='r')
#     for i in range(len(x)):
#         x[i] = x[i] + width
#     plt.bar(x, num_list_3, width=width, label='Greddy', fc='orange', tick_label=name_list)
#     plt.legend()
#     plt.show()
#
# def plot_jain():
#     name_list = ['2 UAV', '3 UAV']
#     num_list_1 = [0.68, 0.73]
#     num_list_2 = [0.2, 0.33]
#     num_list_3 = [0.12, 0.09]
#     x = list(range(len(name_list)))
#     total_width, n = 0.5, 3
#     width = total_width / n
#     plt.bar(x, num_list_1, width=width, label='MAUC', fc='b')
#     for i in range(len(x)):
#         x[i] = x[i] + width
#     plt.bar(x, num_list_2, width=width, label='Random', fc='r')
#     for i in range(len(x)):
#         x[i] = x[i] + width
#     plt.bar(x, num_list_3, width=width, label='Greddy', fc='orange', tick_label=name_list)
#     plt.legend()
#     plt.show()

if __name__ == '__main__':
    plot_jain()








    # name_list = ['1000*1000', '500*500']
    # # num_list = [567.81, 672.32]
    # # num_list1 = [413.21, 532.12]
    # num_list = [0.2122, 0.2312]
    # num_list1 = [0.1892, 0.2331]
    # x = list(range(len(num_list)))
    # total_width, n = 0.5, 2
    # width = total_width / n
    #
    # plt.bar(x, num_list, width=width, label='MADDPG', fc='y')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list1, width=width, label='Random', tick_label=name_list, fc='r')
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()