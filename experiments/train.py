import argparse  # 命令行选项、参数和子命令解释器
import numpy as np
import tensorflow as tf
import time
import pickle
from multiagent.core import Entity
import matplotlib.pyplot as plt

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers  # 包含优化器、正则化、初始化、提供函数模块


def parse_args():
    # 创建一个解释器
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # 添加参数
    # Environment
    parser.add_argument("--scenario", type=str, default="custom_env_extend", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()  # 解析参数


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # 定义agnets的网络结构，三层全连接层，有64个输出单元，激活函数是relu，最后一层没有激活函数（因为类似于回归任务，输出是各个动作的概率值）
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark):
    # 调用MPE环境
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create multiagent environment
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            scenario.benchmark_data, scenario.is_done)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None,
                            scenario.is_done, scenario.cover)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    # 给每个智能体（训练者和对手）定义相关的标号、训练模型、状态集、动作集、arglist
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    '''
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))  # 对手的策略
    '''
    for i in range(num_adversaries, env.n):  # env.n为智能体个数
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers  # trainers可调用MADDPGAgentTrainer类里的方法


def train(arglist):
    with U.single_threaded_session():  # 创建一个会话
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]  # [(8,),(8,),(8,),(8,)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)  # 从/tmp/policy加载训练结果

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []
        # final_ep_rewards = np.array([])  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()  # 重置状态集
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        entity = Entity()
        step_cover = []
        ep_cover = []
        while True:
            # 记录轨迹
            for i in range(env.n):
                entity.path[i].append(obs_n[i][:2])
            # get action,只通过当前智能体自身的观测选择动作
            # 4个智能体有4个trainer（MADDPG）,每个trainer里有2个主网络
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            action_n = np.array(action_n)  # shape=（agents_num, action_num）
            # environment step
            new_obs_n, rew_n, done_n, info_n, cover = env.step(action_n)  # rew_n是有3个元素的list,每个智能体的得分相同
            step_cover.append(cover[0])

            episode_step += 1
            done = True if True in done_n else False
            # done = all(done_n)  # 若done_n都为True则返回True
            terminal = (episode_step >= arglist.max_episode_len)  # max_episode_len=25
            # collect experience of all agents
            # enumerate()将trainers组成一个索引序列，利用它可以同时获得索引和值
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)  # 添加入记忆库
            obs_n = new_obs_n  # 用新的状态信息更新原来的状态信息

            # 累积一个episode的reward值，和agent的reward值
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew  # 3个智能体的reward相加
                agent_rewards[i][-1] += rew  # rew=-10.2

            # 结束标志：done，达到训练步数，则重置环境
            if done or terminal:
                if step_cover[-1] >= 90:
                    plt.figure()
                    plt.plot(np.arange(len(step_cover)), step_cover)
                    plt.ylabel('max_coverage')
                    plt.xlabel('steps')
                ep_cover.append(step_cover[-1])

                np.save("../path_data/trajectory1.npy", np.array(entity.path[0]))
                np.save("../path_data/trajectory2.npy", np.array(entity.path[1]))
                np.save("../path_data/trajectory3.npy", np.array(entity.path[2]))
                np.save("../path_data/trajectory4.npy", np.array(entity.path[3]))
                # print("save trajectory.npy done")

                obs_n = env.reset()
                entity.path = [[] for _ in range(4)]
                episode_step = 0
                episode_rewards.append(0)
                step_cover = []
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1
            # for i in range(env.n):
                # print("the path of %dth UAV is" % i, entity.path[i])

            # for benchmarking learned policies
            # 如果是benchmark==True，就对learned policy进行测试，然后存储数据
            if arglist.benchmark:  # 默认False
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])  # ？
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)  # 将对象agent_info里除了最后一个元素外的所有元素保存到文件fp
                    break
                continue  # 跳过当前循环的剩余语句，然后继续进行下一轮循环

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):  # 每执行完1000个episode
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}, cover: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time()-t_start, 3), ep_cover[-1]))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))  # 记录最后1000个的episode_reward的均值
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))  # 记录最后1000个的agent_reward的均值

            # saves final episode reward for plotting training curve later
            # if len(episode_rewards) > 1000:
            if len(episode_rewards) > arglist.num_episodes:
                final_ep_rewards = np.array(final_ep_rewards)
                final_ep_ag_rewards = np.array(final_ep_ag_rewards)
                np.save("../rewards_data/ep_rewards.npy", final_ep_rewards)
                print("save ep_rewards.npy done")
                np.save("../rewards_data/ag_rewards.npy", final_ep_ag_rewards)
                print("save ag_rewards.npy done")
                """
                rew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)

                agrew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                """
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
