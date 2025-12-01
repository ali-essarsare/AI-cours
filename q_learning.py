import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    # Q-learning update:
    # Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
    best_next = np.max(Q[sprime, :])
    td_target = r + gamma * best_next
    td_error = td_target - Q[s, a]
    Q[s, a] = Q[s, a] + alpha * td_error

    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as input the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    n_actions = Q.shape[1]

    # Exploration
    if np.random.rand() < epsilone:
        return np.random.randint(n_actions)

    # Exploitation: choose best action for state s
    return int(np.argmax(Q[s, :]))


if __name__ == "__main__":
    # You can switch to render_mode=None during training to go faster,
    # and keep "human" only for evaluation.
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1         # learning rate
    gamma = 0.99        # discount factor
    epsilon = 0.2       # initial exploration rate
    epsilon_min = 0.01  # minimal epsilon
    epsilon_decay = 0.99  # decay per episode

    n_epochs = 2000         # number of episodes
    max_itr_per_epoch = 200 # max steps per episode
    rewards = []

    for e in range(n_epochs):
        r = 0.0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            # Choose action with epsilon-greedy policy
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            # Step in the environment
            Sprime, R, terminated, truncated, info = env.step(A)
            done = terminated or truncated

            r += R

            # Q-learning update
            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Move to next state
            S = Sprime

            # Stopping criterion: end of episode
            if done:
                break

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            if epsilon < epsilon_min:
                epsilon = epsilon_min

        print("episode #", e, " : r = ", r)
        rewards.append(r)

    print("Average reward over training = ", np.mean(rewards))

    # Plot the rewards as a function of episodes
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward per episode")
    plt.title("Q-learning on Taxi-v3")
    plt.grid(True)
    plt.show()

    print("Training finished.\n")

    """
    Evaluate the q-learning algorithm
    """

    # Evaluation: run a few episodes using greedy policy (epsilon = 0)
    n_eval_episodes = 5
    for ep in range(n_eval_episodes):
        S, _ = env.reset()
        total_r = 0
        print(f"\nEvaluation episode {ep + 1}")
        for t in range(max_itr_per_epoch):
            # Greedy action (no exploration)
            A = int(np.argmax(Q[S, :]))
            Sprime, R, terminated, truncated, info = env.step(A)
            total_r += R
            S = Sprime
            if terminated or truncated:
                break
        print(f"Total reward (evaluation) = {total_r}")

    env.close()
