import numpy as np
import pickle
from monster_hunt_env import MonsterHuntEnv

env = MonsterHuntEnv()
episodes = 500
alpha = 0.2
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

Q = {}

def get_state(obs):
    grid = obs.reshape(5, 5)
    agent_pos = np.argwhere(grid == 3)[0]
    sword_pos = np.argwhere(grid == 1)[0] if 1 in grid else (-1, -1)
    monsters_pos = np.argwhere(grid == 2)
    has_sword = 1 if env.has_sword else 0
    # Состояние: позиция агента, меча, монстров + здоровье агента + есть ли меч
    state = (*agent_pos, *sword_pos, *monsters_pos.flatten(), env.agent_health, has_sword)
    return tuple(state)

for episode in range(episodes):
    print(f"Episode {episode}")
    obs, _ = env.reset()
    state = get_state(obs)
    done = False
    total_reward = 0
    step_num = 0
    while not done:
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1, 2, 3, 4, 5])
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
        step_num+=1
        done = terminated or truncated or step_num == 20
        total_reward += reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 500 == 0:
        print(f"Episode {episode}, total_reward: {total_reward:.2f}, epsilon: {epsilon:.2f}")

# Сохраняем Q-таблицу
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)
print("Q-table saved to q_table.pkl")
