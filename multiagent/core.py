import numpy as np
import random
import copy


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # trajectory
        self.path = [[]for _ in range(4)]
        # name 
        self.name = ''  # 在具体场景里被重写
        # radius for agent:
        self.size = 10
        # the length of traps
        self.rec_len = 25
        # set vertex for traps(rectangular)
        x = random.choice(range(150, 550, 50))
        y = random.choice(range(150, 550, 50))
        l, r, tp, b = x + self.rec_len / 2, x + self.rec_len / 2 + self.rec_len, y + self.rec_len / 2 + self.rec_len, y + self.rec_len / 2
        self.vertex = [(l, b), (l, tp), (r, tp), (r, b)]
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None  # 在reset_world里重置
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


# properties of landmark entities
class Trap(Entity):
     def __init__(self):
        super(Trap, self).__init__()
        self.size = 25/2


# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # the radius of agents
        # self.size = 10
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.entity = Entity()
        self.agents = []
        self.traps = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.traps

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:  # Agent()
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)  # [None, None, ....]
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)  # 给每个智能体动作赋值(action.u + noise), noise = 0
        # apply environment forces
        # p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)  # 移动智能体
        # for i, agent in enumerate(self.agents):
            # print("the path of %dth UAV is"% i, agent.path[i])
        # update agent state
        # for agent in self.agents:
            # self.update_agent_state(agent)  # 通信动作＋noise

    # 将动作加上噪声
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force
    '''
    # 将agent的运动加在状态上，需要通过get_collision_force判断是不是碰撞
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if(b <= a):
                    continue  # 跳过和自身以及之前重复的
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)  # 判断是否相撞,每撞[0,0]
                if(f_a is not None):
                    if(p_force[a] is None):
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force
    '''
    # 求积分 更改 将速度增加在 P上
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            # origin = copy.deepcopy(entity.state.p_pos)
            # entity.path[i].append(origin)
            # pos = entity.state.p_pos.copy()
            if not entity.movable:
                continue
            '''
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)  # 给智能体减速减速
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt  # 以0.1秒更新, (0.5,0)
            if entity.max_speed is not None:  # entity.max_speed = None
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            # entity.state.p_pos += entity.state.p_vel * self.dt
            '''
            # pos += entity.state.p_vel * 10
            entity.state.p_pos[0] = np.clip(entity.state.p_pos[0]+p_force[i][0], 125, 575)
            entity.state.p_pos[1] = np.clip(entity.state.p_pos[1]+p_force[i][1], 125, 575)
            # entity.path[i].append(entity.state.p_pos)
            # if 100 < pos[0] < 600 and 100 < pos[1] < 600:
                # entity.state.p_pos = pos
            # entity.state.p_pos += entity.state.p_vel * 10

    # 将通信交流动作加上噪声
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):  # entity_b可能是智能体也可能是障碍物
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (entity_a is entity_b):  # 障碍物之间不会相撞
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))  # 智能体之间或智能体与障碍物之间的直线距离
        # minimum allowable distance
        # dist_min = entity_a.size + entity_b.size
        dist_min = 50
        # softmax penetration 多分类器
        k = self.contact_margin  # 1e-3
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k  # log(exp(x1)+exp(x2))
        force = self.contact_force * delta_pos / dist * penetration  # 没有相撞则为0
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]