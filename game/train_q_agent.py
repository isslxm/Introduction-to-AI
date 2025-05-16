import numpy as np
import pickle
from monster_hunt_env import MonsterHuntEnv

env = MonsterHuntEnv()
episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

Q = {}

def get_state(obs):
    return tuple(obs.astype(np.int8).flatten())

for episode in range(episodes):
    obs, _ = env.reset()
    state = get_state(obs)
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            Q.setdefault(state, np.zeros(env.action_space.n))
            action = np.argmax(Q[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = get_state(next_obs)

        if state not in Q:
            Q[state] = np.zeros(env.action_space.n)
        if next_state not in Q:
            Q[next_state] = np.zeros(env.action_space.n)

        # Q-обновление
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

        state = next_state
        done = terminated or truncated
        total_reward += reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 500 == 0:
        print(f"Episode {episode}, total_reward: {total_reward:.2f}, epsilon: {epsilon:.2f}")

# Сохраняем Q-таблицу
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)
print("Q-table saved to q_table.pkl")
