from cgitb import small
from typing import Callable
from matplotlib.pyplot import grid
from tqdm import tqdm
import numpy as np

from model import Model, Actions

#max_iter prevent from going round and round

def sarsa(model: Model,max_iter: int = 100, n_episode: int = 100, epsilon=0.01,alpha=0.01):
    Q = np.zeros((model.num_states, len(Actions)))

    for episode in range(n_episode):
        s = model.start_state
        coin = np.random.choice([0, 1], p=[1-epsilon, epsilon])
        if coin:
            a=np.random.randint(0, len(Actions))
        else:
            a= np.argmax(
                Q[s]
            )
        
        for i in range(max_iter):
            s_ = model.cell2state(model._result_action(model.state2cell(s), a))
            coin = np.random.choice([0, 1], p=[1-epsilon, epsilon])
            if coin:
                a_=np.random.randint(0, len(Actions))
            else:
                a_= np.argmax(
                    Q[s_]
                )
            Q[s][a]=Q[s][a]+alpha*(model.reward(s,a)+model.gamma*Q[s_,a_]-Q[s][a])
            s, a = s_, a_
            if s==model.goal_state:
                break
    V=np.max(Q,axis=1)
    pi=np.argmax(Q)
    return V,pi


import matplotlib.pyplot as plt
from world_config import cliff_world, small_world, grid_world
from plot_vp import plot_vp

model = Model(cliff_world)
V, pi = sarsa(model)
plot_vp(model, V, pi)
plt.show()
