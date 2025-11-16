import numpy as np
from grid import GridWorld

def policy_evaluation(policy, env, theta=1e-4):
    V = np.zeros(env.n * env.n)

    while True:
        delta = 0
        V_new = V.copy()

        for s in env.states:
            if s in env.terminal_states:
                continue

            a = policy[s]
            next_s, r = env.step(s, a)
            V_new[s] = r + env.gamma * V[next_s]

            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < theta:
            break

    return V


def policy_iteration(n=4, gamma=1.0):
    env = GridWorld(n, gamma)

    policy = np.zeros(n*n, dtype=int)  # politique initiale 

    while True:
        V = policy_evaluation(policy, env)
        policy_stable = True

        for s in env.states:
            if s in env.terminal_states:
                continue

            old_action = policy[s]

            actions_values = []
            for a in env.actions:
                next_s, r = env.step(s, a)
                actions_values.append(r + gamma * V[next_s])

            policy[s] = np.argmax(actions_values)

            if old_action != policy[s]:
                policy_stable = False

        if policy_stable:
            break

    return V, policy


if __name__ == "__main__":
    for gamma in [ 0.9, 0.8]:
        V, policy = policy_iteration(gamma=gamma)
        print("Gamma =", gamma)
        print("Value function:\n", V.reshape(4,4))
        print("Policy:\n", policy.reshape(4,4))