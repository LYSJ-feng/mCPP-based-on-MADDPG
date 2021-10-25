import copy

import numpy as np
import random
from multiagent.core import World, Agent, Trap, Entity
from multiagent.scenario import BaseScenario  # 定义了make_world和reset_world方法


class Scenario(BaseScenario):  # 重写BaseScenario中的两个方法
    # 初始化传给了MultiAgentEnv里的world
    def make_world(self):
        world = World()
        # 设置World中的初始化参数
        world.dim_c = 2  # 通信动作的维度？
        # self.area = np.zeros((10, 10))
        self.num_agents = 4   # 无人机个数
        # self.num_trap = 2     # 障碍物个数
        world.collaborative = True
        world.discrete_action = True
        # 向World中的agent列表添加Agent()初始化内容
        # Agent()和Trap()是core.py中Entity()的子类
        world.agents = [Agent() for i in range(self.num_agents)]
        # 重写Agent(Entity)里的内容
        for i, agent in enumerate(world.agents):
            agent.name = 'UAV %d' % i
        # 向World中的trap列表添加Trap()初始化内容
        '''
        world.traps = [Trap() for i in range(self.num_trap)]
        for i, trap in enumerate(world.traps):
            trap.name = 'trap %d' % i
            trap.collide = False
            trap.movable = False
        '''
        # 将上面定义的agent和trap用于reset world
        self.reset_world(world)
        return world

    # 重置环境
    def reset_world(self, world):
        entity = Entity()
        color = [[1, 0.9, 0], [0.35, 0.35, 0.85], [0, 1, 0.9], [0.9, 0, 1]]  # 黄色，紫色，绿色，红色
        for i, agent in enumerate(world.agents):
            agent.color = np.array(color[i])
        # set random initial position states
        lines = [[125 + 50 * random.randint(0, 8), 125], [575, 125 + 50 * random.randint(0, 8)],
                 [175 + 50 * random.randint(0, 8), 575], [125, 175 + 50 * random.randint(0, 8)]]
        for i in range(self.num_agents):
            world.agents[i].state.p_pos = np.array(lines[i % 4], dtype=float)
        for agent in world.agents:
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        self.area = np.zeros((10, 10))
        '''
        for i, trap in enumerate(world.traps):
            ent = Entity()
            trap.state.p_pos = np.array(([ent.vertex[0][0] + 12.5, ent.vertex[0][1] + 12.5]))
            trap.state.p_vel = np.zeros(world.dim_p)
        '''
    def observation(self, agent, world):  # 返回每个智能体的观测状态
        # get positions of all entities in this agent's reference frame
        """
        entity_pos = []  # 无人机与障碍物的距离
        for entity in world.traps:  # [Trap(), Trap(), ...]
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)  # 和智能体的距离
        # entity colors
        entity_color = []
        for entity in world.traps:
            entity_color.append(entity.color)
        """
        # communication of all other agents
        # comm = []
        # x = round((575 - agent.state.p_pos[1]) / 50)
        # y = round((agent.state.p_pos[0] - 125) / 50)
        # idx = self.area[x, y]
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            # comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)  # 和与之通信智能体之间的距离
        # 观测空间包含([速度，智能体位置，智能体和目标距离，与其他智能体的距离，是否通信])
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)  # size = 10
        return np.concatenate([agent.state.p_pos] + other_pos)
        # 每个智能体的状态空间[智能体的横坐标，纵坐标，与其他智能体delta_x, delta_y, 是否被覆盖标记]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each trap, penalized for collisions
        # 惩罚: collision, falling in the traps
        # 奖励: 覆盖率
        rew = 0
        '''
        for l in world.traps:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]  
            min_dist = min(dists)
            if 0 <= min_dist < 50:
                rew -= 1
        '''
        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 1

        for a in world.agents:
            x = round((575 - a.state.p_pos[1]) / 50)
            y = round((a.state.p_pos[0] - 125) / 50)
            self.area[x, y] = 1
        cover = np.sum(self.area == 1)/100
        rew -= 1 - cover

        return rew

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # dist_min = agent1.size + agent2.size
        dist_min = 50
        return True if dist < dist_min else False

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        fall_traps = 0
        '''
        for l in world.traps:
            for a in world.agents:
                dist = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                if 0 <= dist < 50:
                    rew -= 1
                    fall_traps += 1
        '''
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        for a in world.agents:
            x = (575 - a.state.p_pos[1]) / 50
            y = (a.state.p_pos[0] - 125) / 50
            self.area[x, y] = 1
        cover = np.sum(self.area == 1)/100
        rew -= 1-cover
        return (rew, collisions, cover)  # min_dist改为coverage

    def is_done(self):
        return True if np.sum(self.area == 1) == 100 else False

    def cover(self):
        c = copy.deepcopy(np.sum(self.area == 1)/100)
        return c







