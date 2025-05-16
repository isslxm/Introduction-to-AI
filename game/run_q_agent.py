import pickle
import numpy as np
from monster_hunt_env import MonsterHuntEnv

with open(".\game\q_table.pkl", "rb") as f:
    Q = pickle.load(f)

env = MonsterHuntEnv(render_mode="pygame")
obs, _ = env.reset()
done = False

def get_state(obs):
    return tuple(obs.tolist())

while not done:
    env.render()
    state = get_state(obs)
    if state not in Q:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
