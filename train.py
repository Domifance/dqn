import random
import numpy as np
import tensorflow as tf
import gym
from dqn import DEEPQ
from network import get_network_builder
from replay_buffer import ReplayBuffer,PrioritizedReplayBuffer
from schedules import LinearSchedule

# env = gym.make("Breakout-v4")
# import ale_py
# from ale_py import ALEInterface
# ale = ALEInterface()
# from ale_py.roms import Breakout
# ale.loadROM(Breakout)
# env = gym.make("Breakout-v4")
env = gym.make("CartPole-v1")
env.render()

gamma=0.99  # discounting factor
max_grad_norm=0.5  # gradient norm clipping coefficient
cliprange=0.2   # clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training and 0 is the end of the training
lr=3e-4    # learning rate , onstant or schedule function [0,1] -> R+ where 1 is beginning of the training and 0 is the end of the training
episode = 1000   # 总共训练多少场比赛
save_episode = 20   # 多少场 保存一次 参数
param_noise = False
load_path = None
prioritized_replay = False


print(env.observation_space.shape)
print(env.action_space.n)
obs_shape = env.observation_space.shape    # 状态空间 形状
acition_n = env.action_space.n             # 动作空间 形状
# q_network_fn = get_network_builder("impala")()
q_network_fn = get_network_builder("mlp")()


buffer_size = 10000     # replay缓存大小
episode_rewards = []    # 每个 episode 的 回报
batch_size = 32
learning_starts = 20
target_network_update_freq = 10
train_freq = 2
episode_train = 100
exploration_final_eps = 0.01
# network = q_network_fn(encode_obs_shape)
# 执行模型，获取动作 和价值
# run_model = PolicyWithValue(action_n,network)

# 训练模型，用于训练 并保存模型参数
dqn_model = DEEPQ(
        q_func=q_network_fn,
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        lr=lr,
        grad_norm_clipping=10,
        gamma=gamma,
        param_noise=param_noise
    )
# a = input()
# 加载模型
if load_path is not None:
    # load_path = osp.expanduser(load_path)
    ckpt = tf.train.Checkpoint(model=dqn_model)
    manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
    ckpt.restore(manager.latest_checkpoint)
    print("Restoring from {}".format(manager.latest_checkpoint))

# Create the schedule for exploration starting from 1.
exploration = LinearSchedule(schedule_timesteps=int(episode * batch_size * episode_train),
                                initial_p=1.0,
                                final_p=exploration_final_eps)

# Create the replay buffer
# if prioritized_replay:
#     replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
#     if prioritized_replay_beta_iters is None:
#         prioritized_replay_beta_iters = total_timesteps
#     beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
#                                     initial_p=prioritized_replay_beta0,
#                                     final_p=1.0)
# else:
#     replay_buffer = ReplayBuffer(buffer_size)
#     beta_schedule = None
replay_buffer = ReplayBuffer(buffer_size)

dqn_model.update_target()


for epi in range(episode):
    """一场比赛，先收集数据， 后训练"""
    
    # 重置环境 
    obs = env.reset()
    print("the episode:{}".format(epi))
    total_reward = 0
    update_eps = tf.constant(exploration.value(episode))   # 更新 epsilon
    kwargs = {}

    # exploration
    print("exploration")
    for i in range(batch_size*episode_train):
        action, _, _, _ = dqn_model.step(tf.constant([obs]), update_eps=update_eps, **kwargs)
        action = action[0].numpy()

        # print(action)
        new_obs, rew, done, _ = env.step(action)
        # print(new_obs)
        # print(np.sum(new_obs))
        # print(done)

        total_reward += rew
        # Store transition in the replay buffer.
        if done:
            # print("done: ",done)
            # a = input()
            obs = env.reset()
            episode_rewards.append(total_reward)
            total_reward = 0
        else:
            obs = new_obs
        replay_buffer.add(obs, action, rew, new_obs, float(done))
    
    print("training ...")
    for j in range(episode_train):
        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
        actions, rewards, dones = tf.constant(actions), tf.constant(rewards), tf.constant(dones)
        weights, batch_idxes = np.ones_like(rewards), None
        td_errors = dqn_model.train(obses_t, actions, rewards, obses_tp1, dones, weights)  # 训练

    dqn_model.update_target()

    print(episode_rewards)
    if len(episode_rewards) > 0:
        mean_10ep_reward = round(np.mean(episode_rewards[-11:-1]), 1)
        print("mean 10 episode reward :{}".format(mean_10ep_reward))

# 关闭环境
env.close()   # 关闭环境

