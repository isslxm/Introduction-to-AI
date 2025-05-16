import pickle
import numpy as np
import time
from monster_hunt_env import MonsterHuntEnv
import pygame

pygame.init()

with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

env = MonsterHuntEnv(render_mode="pygame")
obs, _ = env.reset()
done = False

# Создаём окно Pygame
env.window = pygame.display.set_mode((env.window_size, env.window_size))
pygame.display.set_caption("Monster Hunt")

# Теперь можно создавать шрифт (после pygame.init())
env.font = pygame.font.Font(None, 24)

# Загрузка и масштабирование изображений
env.agent_img = pygame.image.load("assets/knight2.png").convert_alpha()
env.agent_img = pygame.transform.scale(env.agent_img, (env.cell_size, env.cell_size))

env.sword_img = pygame.image.load("assets/sword.png").convert_alpha()
env.sword_img = pygame.transform.scale(env.sword_img, (env.cell_size // 2, env.cell_size // 2))

env.monster_img = pygame.image.load("assets/monster3.png").convert_alpha()
env.monster_img = pygame.transform.scale(env.monster_img, (env.cell_size, env.cell_size))

def get_state(obs):
    return tuple(obs.tolist())

episodes = 5

for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    print(f"=== Episode {episode+1} ===")
    while not done:
        env.render()
        time.sleep(0.3)
        state = get_state(obs)
        if state not in Q:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

pygame.quit()