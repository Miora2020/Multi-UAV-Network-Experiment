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
import pickle
import matplotlib.pyplot as plt
import random

learning_rate = 0.001
learning_start_step = int(1e5)  # 开始训练的游戏步数
learning_fre = 100  # 100
batch_size = 1280  # 2500
gamma = 0.9  # 折扣系数
save_frequency = 1000  # 2000
save_dir = './models'
tao = 0.01
Num_hidden_1 = 128
Num_hidden_2 = 64
BUFFER_SIZE = int(1e5)  # 5000
max_episode = 100000   # 1000
per_episode_max_len = 200
max_grad_norm = 0.5
var = 0.1
alpha = 0.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weigth_init(m):
   if isinstance(m, nn.Conv2d):
       init.xavier_uniform_(m.weight.data)
       init.constant_(m.bias.data,0.1)
   elif isinstance(m, nn.BatchNorm2d):
       m.weight.data.fill_(1)
       m.bias.data.zero_()
   elif isinstance(m, nn.Linear):
       m.weight.data.normal_(0, 0.1)
       m.bias.data.zero_()


def get_trainers(env, obs_shape_n, action_shape_n, num_hidden_1, num_hidden_2):
    '''
    新建trainer或加载之前的模型
    :param env:
    :param obs_shape_n:
    :param action_shape_n:
    :return:
    '''
    actors_cur = [None for _ in range(env.world.num_UAVs)]
    critics_cur = [None for _ in range(env.world.num_UAVs)]
    actors_tar = [None for _ in range(env.world.num_UAVs)]
    critics_tar = [None for _ in range(env.world.num_UAVs)]
    optimizers_c = [None for _ in range(env.world.num_UAVs)]  # 优化器
    optimizers_a = [None for _ in range(env.world.num_UAVs)]

    for i in range(env.world.num_UAVs):
        actors_cur[i] = actor_agent(obs_shape_n[i], num_hidden_1, num_hidden_2, action_shape_n[i]).to(device)
        actors_tar[i] = actor_agent(obs_shape_n[i], num_hidden_1, num_hidden_2, action_shape_n[i]).to(device)
        critics_cur[i] = critic_agent(sum(obs_shape_n), num_hidden_1, num_hidden_2, sum(action_shape_n)).to(device)
        critics_tar[i] = critic_agent(sum(obs_shape_n), num_hidden_1, num_hidden_2, sum(action_shape_n)).to(device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), lr=learning_rate)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), lr=learning_rate)
        # actors_cur[i].apply(weigth_init)
        # critics_cur[i].apply(weigth_init)

    actors_tar = update_trainers(actors_cur, actors_tar, tao) # 1.0 update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, tao) # 1.0 update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def update_trainers(agents_cur, agents_tar, tao):
    for agent_cur, agent_tar in zip(agents_cur, agents_tar):
        key_list = list(agent_cur.state_dict().keys())
        state_dict_t = agent_tar.state_dict()
        state_dict_c = agent_cur.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agent_tar.load_state_dict(state_dict_t)
    return agents_tar


def agents_train(episode, game_step, update_cnt, memory, obs_size, action_size, \
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    '''
    训练智能体的actor和critic网络
    :param game_step: 当前回合的游戏步数
    :param update_cnt: 网络的更新次数
    :param memory: 经验池
    :param obs_size: 观察空间维度
    :param action_size: 动作空间维度
    :param actors_cur:
    :param actors_tar:
    :param critics_cur:
    :param critics_tar:
    :param optimizers_a:
    :param optimizers_c:
    :return:
    '''
    # ??为什么要设置learning_start_step?
    if game_step > learning_start_step and (game_step - learning_start_step) % learning_fre == 0:
        if update_cnt == 0:
            print('\r=start training ...'+' '*100)
        # 更新网络
        update_cnt += 1  # 更新次数+1
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) \
            in enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            # 从memory中采样
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample(batch_size, agent_idx)  # Note_The func is not the same as others  o指的是old，n表示new

            # 更新critic网络
            rew = torch.tensor(_rew_n, dtype=torch.float, device=device)
            done = torch.tensor(~_done_n, dtype=torch.float, device=device)  # 取反，并变为0/1
            action_cur_o = torch.from_numpy(_action_n).to(device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(device, torch.float)
            # tar网络输出的动作
            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # 计算q值
            q_target = critic_t(obs_n_n, action_tar).reshape(-1)  # 使用目标网络计算目标值
            # 有多个样本，所以这里的done有多个
            # print(rew)
            # print(done)
            # print(q)
            # print(q_target)
            tar_value = rew + gamma * q_target * done
            loss_c = nn.MSELoss()(q, tar_value)
            opt_c.zero_grad()
            loss_c.backward()
            # 是否进行梯度剪裁？
            nn.utils.clip_grad_norm_(critic_c.parameters(), max_grad_norm)
            opt_c.step()

            # # openai架构的actor网络更新
            # model_out, policy_new = actor_c(obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_new
            # loss_pse = torch.mean(torch.pow(model_out, 2))
            # loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))
            # opt_a.zero_grad()
            # # (1e-3 * loss_pse + loss_a).backward()
            # loss_a.backward()
            # nn.utils.clip_grad_norm_(actor_c.parameters(), max_grad_norm)
            # opt_a.step()

            # 基础模型的actor网络更新
            policy_new = actor_c(obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]])
            # 将新动作插入到就动作中
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_new
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))
            opt_a.zero_grad()
            loss_a.backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), max_grad_norm)
            opt_a.step()

        # 保存模型
        if update_cnt >= save_frequency and update_cnt % save_frequency == 0:
            model_file_dir = os.path.join(save_dir, '{}UAVs_{}'.format(len(actors_cur), episode))
            if not os.path.exists(model_file_dir):
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t{}.pt'.format(agent_idx)))

        # 更新tar网络
        actors_tar = update_trainers(actors_cur, actors_tar, tao)
        critics_tar = update_trainers(critics_cur, critics_tar, tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

def train(var):
    # 创建环境
    scenario = sc()
    env = MultiUAVEnv(scenario)
    # env.render()
    # 创建agents
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.world.num_UAVs)]
    action_shape_n = [env.action_space[i].shape[0] for i in range(env.world.num_UAVs)]
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, obs_shape_n, action_shape_n, Num_hidden_1, Num_hidden_2)
    memory = ReplayBuffer(BUFFER_SIZE)

    # 初始化网络参数
    obs_size = []
    action_size = []
    game_step = 0   # game_step 是一个全局的步数，完成一个episode时，game_step不清零
    update_cnt = 0
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.world.num_UAVs)]
    jain_Index = []
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    times = []
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)  # obs_size中的range存储了每个agent的局部观察对应的整个全局观察的开始和结束下标
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a
    print(obs_shape_n)
    print(obs_size)
    print(action_size)

    # 开始进行迭代训练
    obs_n = env.reset()

    for episode_gone in range(max_episode):
        duration = 0
        while True:
            # get action
            action_n = None
            if game_step < learning_start_step:
                action_n = env.random_action()
            else:
                if random.random() < alpha:
                    action_n = env.random_action()
                else:
                    action_n = [
                        agent(torch.from_numpy(obs).to(device, torch.float)).detach().cpu().numpy() + np.random.randn(2)*var for agent, obs in zip(actors_cur, obs_n)]
            # 和环境交互
            new_obs_n, reward, done_n, info_n = env.step(action_n)
            # print(new_obs_n)
            # 保存到经验池
            memory.add(obs_n, np.concatenate(action_n), reward, new_obs_n, done_n)
            episode_rewards[-1] += np.sum(reward)
            for i, rew in enumerate(reward):
                agent_rewards[i][-1] += reward[i]

            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(episode_gone, game_step, update_cnt, memory, obs_size, action_size, actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)
            # 更新噪声参数
            if game_step > learning_start_step and var > 0.05:
                var *= 0.999998
            game_step += 1
            obs_n = new_obs_n
            done = False
            if True in done_n:
                done = True
            if done or env.world.t >= 200:  # world.t大于200时
                duration = env.world.t
                times.append(duration)
                tmp = env.get_Jain_Index()
                fairness.append(tmp)
                for landmark in env.world.landmarks.values():
                    tmp.append(landmark.sum_throughput)
                    fairness.append(tmp)
                obs_n = env.reset()
                for r in agent_rewards:
                    r.append(0)
                episode_rewards.append(0)
                break
        print('=Training: steps:{} episode:{} duration:{}'.format(game_step, episode_gone, duration))
        mean_ep_r = round(episode_rewards[-2], 3)
        mean_agents_r = [round(agent_rewards[idx][-2], 2) for idx in range(env.world.num_UAVs)]
        print(" " * 43 + 'episode reward:{}    agent_rewards:{}'.format(mean_ep_r, mean_agents_r))
        print(var)

        if episode_gone % 500 == 0:
            x = [i for i in range(len(episode_rewards)-1)]
            plt.plot(x, episode_rewards[:-1])
            plt.xlabel('epoch')
            plt.ylabel('reward')
            plt.savefig('train_results/episode {} reward'.format(episode_gone))
            plt.show()

            y = [i for i in range(len(times))]
            plt.plot(x, times[:])
            plt.xlabel('epoch')
            plt.ylabel('duration')
            plt.savefig('train_results/episode {} duration'.format(episode_gone))
            plt.show()


    with open ("reward.txt", 'wb') as f:
        pickle.dump(episode_rewards, f)
    with open ("duration.txt", "wb") as f:
        pickle.dump(times, f)
    with open ("agent_reward.txt", 'wb') as f:
        pickle.dump(agent_rewards, f)
    with open ("fairness.txt", 'wb') as f:
        pickle.dump(fairness, f)

if __name__ == '__main__':
    train(var)
    # scenario = sc()
    # env = MultiUAVEnv(scenario)
    # # env.render()
    # # 创建agents
    # obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.world.num_UAVs)]
    # action_shape_n = [env.action_space[i].shape[0] for i in range(env.world.num_UAVs)]
    # actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
    #     get_trainers(env, obs_shape_n, action_shape_n, Num_hidden_1, Num_hidden_2)
    # print(actors_cur)
