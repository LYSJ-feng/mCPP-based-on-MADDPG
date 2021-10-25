import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.collections as coll

p1 = np.load('./path_data/trajectory1.npy')
p2 = np.load('./path_data/trajectory2.npy')
p3 = np.load('./path_data/trajectory3.npy')
p4 = np.load('./path_data/trajectory4.npy')
x = []
y = []
'''
for i in range(len(p1)):
    x.append(p1[i][0])
    y.append(p1[i][1])
plt.plot(x, y)
'''
for i in range(len(p1)):
    plt.scatter(p1[i][0], p1[i][1], marker='x', color='red', s=40, label='UAV1')
    plt.scatter(p2[i][0], p1[i][1], marker='+', color='blue', s=45, label='UAV2')
    plt.scatter(p3[i][0], p1[i][1], marker='o', color='green', s=40, label='UAV3')
    plt.scatter(p4[i][0], p1[i][1], marker='^', color='orange', s=40, label='UAV4')

plt.axis([100, 600, 100, 600], 'equal')
plt.xticks(np.arange(100, 650, 50))
plt.yticks(np.arange(100, 650, 50))
plt.grid()
plt.gca().set_aspect("equal")
# plt.legend(loc='best')
plt.show()

