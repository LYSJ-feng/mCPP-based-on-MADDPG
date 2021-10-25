import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):  # make_world，reset_world
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3  # 智能体和目标位数目相同
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]  # 往空列表里添加三个Agent(),包括所有初始化的量
        for i, agent in enumerate(world.agents):  # enumerate()将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据
            agent.name = 'agent %d' % i
            agent.collide = True  # 可能相撞
            agent.silent = True   # 不通信
            agent.size = 0.15     # 圆的半径
        # add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)  # 初始化为World()里的初始化
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)  # 从一个均匀分布[-1,1)中随机采样2个数,位置初始化
            agent.state.p_vel = np.zeros(world.dim_p)  # 速度初始化（0,0）
            agent.state.c = np.zeros(world.dim_c)  # 通信信号初始化（0,0）
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)  #
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]  # 每个智能体和不同目标的距离
            min_dists += min(dists)  # 每个目标与智能体最小距离之和
            rew -= min(dists)        # 距离越小奖励值越高
            if min(dists) < 0.1:
                occupied_landmarks += 1  # 距离小于0.1则认为覆盖到目标点
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))  # 两个智能体的相对距离
        dist_min = agent1.size + agent2.size  # 最小不碰撞距离为两智能体半径之和
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)  # 目标和智能体的相对位置
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue  # 如果通信对象是自身则跳出当前for
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos) # 智能体和通信对象的相对距离
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        # 返回的状态集为array([智能体的速度，智能体的位置，目标地和智能体相对距离，智能体和通信对象相对距离，通信动作])
