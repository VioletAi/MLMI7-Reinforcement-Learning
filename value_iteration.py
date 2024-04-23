from cgitb import small
from typing import Callable
from matplotlib.pyplot import grid
from tqdm import tqdm
import numpy as np

from model import Model, Actions


def value_iteration(model: Model, maxit: int = 100, asynchronous=True):

    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))

    def compute_value(s, a, reward: Callable):
        return np.sum(
            [
                model.transition_probability(s, s_, a)
                * (reward(s, a) + model.gamma * V[s_])
                for s_ in model.states
            ]
        )
    def policy_improvement():
        for s in model.states:
            action_index = np.argmax(
                [compute_value(s, a, model.reward) for a in Actions]
            )
            pi[s] = Actions(action_index)


    delta=1e+6
    delta_0=0.01

    while delta>=delta_0:

        delta=0

        if asynchronous:
            for s in model.states:
                temp=V[s]
                vs = []
                for a in Actions:
                    R = model.reward(s, a)
                    v = compute_value(s, a, lambda *_: R)
                    vs.append(v)
                V[s] = max(vs)
                delta = max(delta, abs(V[s] - temp))
        
        else:

            V_new = np.zeros_like(V)
            for s in model.states:
                temp=V[s]
                vs = []
                for a in Actions:
                    R = model.reward(s, a)
                    v = compute_value(s, a, lambda *_: R)
                    vs.append(v)
                V_new[s] = max(vs)
                delta = max(delta, abs(V_new[s] - temp))
            V = np.copy(V_new)

    policy_improvement()

    return V, pi


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from world_config import cliff_world, small_world, grid_world
    from plot_vp import plot_vp

    model = Model(cliff_world)
    V, pi = value_iteration(model)
    plot_vp(model, V, pi)
    plt.show()
