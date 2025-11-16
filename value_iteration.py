import numpy as np
from grid import GridWorld

def value_iteration(n=4, gamma=1.0, theta=1e-4):
    env = GridWorld(n, gamma)
    V = np.zeros(n*n)

    while True:
        delta = 0
        V_new = V.copy()

        for s in env.states:
            if s in env.terminal_states:
                continue

            values = []
            for a in env.actions:
                next_s, r = env.step(s, a)
                values.append(r + gamma * V[next_s])

            V_new[s] = max(values)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < theta:
            break

    policy = np.zeros(n*n, dtype=int)

    for s in env.states:
        if s in env.terminal_states:
            policy[s] = -1
            continue

        values = []
        for a in env.actions:
            next_s, r = env.step(s, a)
            values.append(r + gamma * V[next_s])

        policy[s] = np.argmax(values)

    return V, policy

if __name__ == "__main__":
    for gamma in [1.0, 0.9, 0.8]:
        V, policy = value_iteration(gamma=gamma)
        print("Gamma =", gamma)
        print("Value function:\n", V.reshape(4,4))
        print("Policy:\n", policy.reshape(4,4))