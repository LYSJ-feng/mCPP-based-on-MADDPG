import gym
from gym import spaces, core
from gym.envs.registration import EnvSpec
import numpy as np
import logging  # 用于输出运行日志
from multiagent.multi_discrete import MultiDiscrete
import random

logger = logging.getLogger(__name__)  # 定义日志接口


class MultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 2}  # 添加元数据，改变渲染环境时的参数

    def __init__(self, world, reset_callback=None, reward_callback=None, observation_callback=None, info_callback=None,
                 done_callback=None, cover_callback=None, viewer=None):
        self.viewer = viewer  # None则可视化
        self.world = world    # scenario.make_world()
        self.agents = self.world.policy_agents
        self.n = len(world.policy_agents)  # 智能体数目
        # scenario callbacks
        self.reset_callback = reset_callback    # scenario.reset_world
        self.reward_callback = reward_callback  # scenario.reward
        self.observation_callback = observation_callback  # scenario.observation
        self.info_callback = info_callback  # scenario.benchmark_data
        self.done_callback = done_callback  # None
        self.cover_callback = cover_callback
        # environment parameters
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_space = True
        # if ture, even the action is continuous, action will be performed discretely
        self.discrete_action_input = False
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False  # True，和discrete_action_input互斥
        # self.shared_reward if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []  # 所有智能体的动作空间
        self.observation_space = []  # size=4*8=32
        for agent in self.agents:
            total_action_space = []  # 每个智能体有一个总的动作空间
            total_observation_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)  # Discrete(5)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)  # Box(2,)代表横纵坐标的数值，每一维区间最小值为-1最大为1
            if agent.movable:
                total_action_space.append(u_action_space)  # Discrete(5) or Box(2,)
            '''
            # communication action space，不通信，不添加进总动作空间
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)  # Discrete(2)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)  # Box(2,)
            if not agent.silent:
                total_action_space.append(c_action_space)  # 添加进总的动作空间
            # total action space
            if len(total_action_space) > 1:  # have both physical action and communication action
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])  # [[0,4],[0,1]]
                else:  
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:  # len(total_action_space)=1
                self.action_space.append(total_action_space[0])  # 只有物理动作时，action_space = [Discrete(5]
            '''
            self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = len(observation_callback(agent, self.world))  # 观测向量的维度，observation_callback在具体环境中定义, obs_dim = 8
            # self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # 连续观测空间?
            ag = np.linspace(125, 575, 10)
            delta = np.linspace(0, 450, 10)
            for a in range(len(agent.state.p_pos)):
                total_observation_space.append(random.choice(ag))
            for b in range(obs_dim-len(agent.state.p_pos)):
                total_observation_space.append(random.choice(delta))
            self.observation_space.append(np.array(total_observation_space))
            agent.action.c = np.zeros(self.world.dim_c)

    def step(self, action_n):  # 输入第i个智能体的实际动作
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        cover_n = []
        self.agents = self.world.policy_agents  # [Agent(), Agent()..]
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])  # 给action_n[i]赋值
        # advance world state in core.py
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))         # 每个智能体的观测状态的集合
            reward_n.append(self._get_reward(agent))   # 每个智能体的奖励集合
            done_n.append(self._get_done(agent))       # return False
            info_n['n'].append(self._get_info(agent))  # 返回奖赏、撞机次数，覆盖率，撞到障碍物的次数
        cover_n.append(self._get_cover(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)  # 奖赏函数为智能体的奖赏总和
        if self.shared_reward:  # True
            reward_n = [reward] * self.n
        # 返回所有agent的观测,[所有agent的reward之和]*4,False,所有agent的信息
        return obs_n, reward_n, done_n, info_n, cover_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    def _get_cover(self, agent):
        if self.cover_callback is None:
            return {}
        return self.cover_callback()

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:  # scenario.observation
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback()

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):  # 如果action_space是多个离散动作
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]  # [第i个智能体的实际动作]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0  # 左
                if action[0] == 2: agent.action.u[0] = +1.0  # 右
                if action[0] == 3: agent.action.u[1] = -1.0  # 下
                if action[0] == 4: agent.action.u[1] = +1.0  # 上
            else:
                if self.force_discrete_action:  # one hot
                    d = np.argmax(action[0])    # 最大概率的位置置1其余0
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:  # True, policy-based
                    agent.action.u[0] += action[0][1] - action[0][2]  # 以one hot形式加减
                    agent.action.u[1] += action[0][3] - action[0][4]  # action[0][0]为静止，[1]右[2]左[3]上[4]下
                else:
                    agent.action.u = action[0]  # 连续动作，输出概率值
            sensitivity = 50
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]  # 把下一类动作(communication action)赋给action
        '''
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)  # ([0,0])
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        '''
        # make sure we used all elements of action
        assert len(action) == 0  # 确认每种动作都config了

    # render environment
    def render(self, mode='human', close=False):
        screen_width = 700
        screen_height = 700
        # create rendering geometry
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)  # 创建长和宽为700*700的画板

            # 创建10*10的网格
            for n in range(11):
                line_x = rendering.Line((100, 100+50 * n), (600, 100+50 * n))
                line_y = rendering.Line((100+50 * n, 100), (100+50 * n, 600))
                line_x.set_color(0, 0, 0)  # 对应rgb,(0,0,0)为黑色
                line_y.set_color(0, 0, 0)
                # 把元素添加到画板中
                self.viewer.add_geom(line_x)
                self.viewer.add_geom(line_y)

            # 随机创建无人机和障碍物位置
            self.render_geoms = []        # 环境元素的集合
            self.render_geoms_xform = []  # 所有元素位置信息的集合
            color = (1, 0.9, 0, 0.35, 0.35, 0.85, 0, 1, 0.9, 0.9, 0, 1)
            for i, agent in enumerate(self.world.agents):
            # for entity in self.world.entities:
                xform = rendering.Transform()
                # if 'UAV' in entity.name:
                uav = rendering.make_circle(10)
                uav.set_color(color[i], color[i+1], color[i+2])
                uav.add_attr(xform)
                self.render_geoms.append(uav)
                '''
                if 'trap' in entity.name:
                    rec = rendering.FilledPolygon(entity.vertex)
                    rec.add_attr(xform)
                    self.render_geoms.append(rec)
                '''
                self.render_geoms_xform.append(xform)

            for geom in self.render_geoms:
                self.viewer.add_geom(geom)  # 将图形添加到画板上

            '''
            # UAV1
            self.uav1 = rendering.make_circle(10)
            self.uav1_trans = rendering.Transform(translation=(125+50*random.randint(0, 8), 125))  # 无人机的初始位置
            self.uav1.add_attr(self.uav1_trans)
            self.uav1.set_color(1, 0.9, 0)

            # UAV2
            self.uav2 = rendering.make_circle(10)
            self.uav2_trans = rendering.Transform(translation=(575, 125 + 50 * random.randint(0, 8)))  # 无人机的初始位置
            self.uav2.add_attr(self.uav2_trans)
            self.uav2.set_color(1, 0.9, 0)

            # UAV3
            self.uav3 = rendering.make_circle(10)
            self.uav3_trans = rendering.Transform(translation=(175 + 50 * random.randint(0, 8), 575))  # 无人机的初始位置
            self.uav3.add_attr(self.uav3_trans)
            self.uav3.set_color(1, 0.9, 0)

            # UAV4
            self.uav4 = rendering.make_circle(10)
            self.uav4_trans = rendering.Transform(translation=(125, 175 + 50 * random.randint(0, 8)))  # 无人机的初始位置
            self.uav4.add_attr(self.uav4_trans)
            self.uav4.set_color(1, 0.9, 0)

            self.viewer.add_geom(self.uav1)
            self.viewer.add_geom(self.uav2)
            self.viewer.add_geom(self.uav3)
            self.viewer.add_geom(self.uav4)
            
            # 创建障碍物位置
            rec_len = 25
            x1 = random.choice(range(150, 550, 50))
            y1 = random.choice(range(150, 550, 50))
            x2 = random.choice(range(150, 550, 50))
            y2 = random.choice(range(150, 550, 50))
            if x1 == x2 and y1 == y2:
                x2 = random.choice(range(150, 550, 50))
                y2 = random.choice(range(150, 550, 50))
            l1, r1, tp1, b1 = x1+rec_len/2, x1+rec_len/2+rec_len, y1+rec_len/2+rec_len, y1+rec_len/2
            self.rec1 = rendering.FilledPolygon([(l1, b1), (l1, tp1), (r1, tp1), (r1, b1)])  # 矩形四个顶点坐标
            l2, r2, tp2, b2 = x2+rec_len/2, x2+rec_len/2+rec_len, y2+rec_len/2+rec_len, y2+rec_len/2
            self.rec2 = rendering.FilledPolygon([(l2, b2), (l2, tp2), (r2, tp2), (r2, b2)])  # 矩形四个顶点坐标
            self.rec1_trans = rendering.Transform()
            self.rec2_trans = rendering.Transform()
            self.rec1.add_attr(self.rec1_trans)
            self.rec2.add_attr(self.rec2_trans)
            self.viewer.add_geom(self.rec1)
            self.viewer.add_geom(self.rec2)
            # 把无人机和障碍物的位置信息打包, 传入state中的位置信息
            self.render_geoms_xform.append(self.uav1)
            self.render_geoms_xform.append(self.uav2)
            self.render_geoms_xform.append(self.uav3)
            self.render_geoms_xform.append(self.uav4)
            self.render_geoms_xform.append(self.rec1)
            self.render_geoms_xform.append(self.rec2)
            # print(uav1_trans.translation[0])
            '''
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)  # 传入新坐标

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n

