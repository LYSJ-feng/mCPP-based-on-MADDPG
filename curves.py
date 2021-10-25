import pickle
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sns.set(style='ticks', rc={'font.sans-serif': 'SimHei', 'axes.unicode_minus': False})   # 用来显示中文
y1 = np.load('./rewards_data/ep_rewards.npy')
y2 = np.load('./rewards_data/ag_rewards.npy')
x1 = range(len(y1))
x2 = range(len(y2))
"""
sns.set(style="darkgrid", font_scale=1.5)
sns.tsplot(time=x1, data=y1, color="r", condition="ep_rewards")
sns.tsplot(time=x2, data=y2, color="b", condition="ag_rewards")
# doc = open('1.txt', 'a')

plt.ylabel("Reward")
plt.xlabel("Iteration Number")
plt.title("MADDPG_simple_spread")

plt.show()
"""
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
ax.plot(x1, y1, color='r')
plt.xlabel('Iteration Number')
plt.ylabel('ep_rewards')

plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
ax.plot(x2, y2, color='b')
# plt.ylim(-800, -400)
plt.xlabel('Iteration Number')
plt.ylabel('ag_rewards')

plt.show()
