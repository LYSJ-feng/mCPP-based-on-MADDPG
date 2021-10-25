import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):  # [::-1]逆序
        r = reward + gamma*r
        r = r*(1.-done)  # done为True时r=0, False时r=r*1
        discounted.append(r)
    return discounted[::-1]  # 逆序后才是时间由先到后的顺序


# 更新target网络参数(软更新方式)
def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2  # 0.99
    expression = []
    # lambda表示匿名函数，可对元组组成的列表中某一项进行排序
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))  # assign()给var_target添加一列
    expression = tf.group(*expression)  # * 的作用是把列表expression中的每个元素，当作位置参数传进去（解压参数列表），如group(1,2,3)
    return U.function([], [], updates=[expression])


# 建立actor,建立 critic,然后将critic的输出和actor输出当做loss来训练actor
# 定义policy（actor）网络结构和训练过程
def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):  # 用于定义创建变量（层）的操作的上下文管理器
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  # 设定action的概率分布类型和概率的值，[5,5,5,5]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]  # p网络的输入为当前无人机的局部状态

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)  # mlp_model((,8), 5)输入当前无人机的状态，输出策略(概率分布)p.shape=(?,5)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))  # 6个网络参数

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()  # 根据P输出的分布act_pd采样,softmax得到动作
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))  # 计算近似的最优解

        act_input_n = act_ph_n + []  # critic网络的输入为4架无人机的动作
        act_input_n[p_index] = act_pd.sample()  # 将p网络输出的动作给相应智能体作为q网络的输入
        q_input = tf.concat(obs_ph_n + act_input_n, 1)  # critic网络的输入所有agent的观测和动作，在1维度concat obs_ph_n和act_input_n=（x',a1,a2,a3...aN）
        # 如果是DDPG，就将local_q_func标记为True，并将该智能体的局部observation和actions传入给critic网络
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)  # tf.concat在1维度上拼接状态与动作
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]  # 定义critic网络（传入actor网络得到的输出动作，全局信息），输出q值
        pg_loss = -tf.reduce_mean(q)  # q值越大，动作的概率越大，则损失越小，所以q值和loss是反相关

        loss = pg_loss + p_reg * 1e-3  # 定义用于训练policy network的损失函数，preg*1e-3为正则项，系数1e-3表示近似最优解p_reg的可信程度

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions(用于输出损失函数、输出动作具体值)
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])  # critic网络输入：全局观测状态和actor网络输出的动作；输出loss,优化器更新
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)  # actor的输入：当前状态；输出：动作
        p_values = U.function([obs_ph_n[p_index]], p)  # 策略值（奖励）：用当前策略下的观测和策略得到

        # target network of policy network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))  # target policy的策略集合
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)  # 更新target actor 网络的参数

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample() # target-network 输出的动作
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)  # 计算target-network的输出值

        # 返回输出的动作,训练loss,网络参数，策略奖励和target网络的输出动作
        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


# 定义q（critic）网络结构和训练过程
def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]  # 设定action的概率分布类型和概率的值

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)  # q网络的输入为全局观测和全局动作，总共大小是52
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss  # + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name  # 所用trainer的name
        self.n = len(obs_shape_n)  # 智能体数目4
        self.agent_index = agent_index  # agent的索引
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())  # Creates a placeholder for a batch of tensors of a given shape

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len  # 1024*25
        self.replay_sample_index = None

    def action(self, obs):  # obs是其中一个智能体的状态集
        return self.act(obs[None])[0]  # obs[None]给obs加了一维，shape=（1，n）

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    # 每100步更新一次replay buffer
    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)  # 采样batch_size大小的数据的编号
        # collect replay sample from all agents
        obs_n = []  # 当前所有智能体的状态
        obs_next_n = []  # 下一时刻所有智能体的状态
        act_n = []  # 所有智能体的动作
        index = self.replay_sample_index  # 采样数据的编号
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))  # target_q_network的输入为下一步的全局观测以及target_p_network的输出动作
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next  # critic网络的目标函数
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))  # 计算critic网络的损失函数（均方误差）

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))  # 计算actor网络的损失函数

        self.p_update()  # 更新policy网络参数
        self.q_update()  # 更新q网络参数

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
