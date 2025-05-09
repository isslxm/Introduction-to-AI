import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
import time

class MonsterHuntEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = 5
        self.cell_size = 100
        self.window_size = self.grid_size * self.cell_size
        self.window = None
        self.render_mode = render_mode
        self.font = None

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=int)

        self.agent_img = None
        self.sword_img = None
        self.monster_img = None

    def _get_random_free_call(self):
        while True:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if self.grid[y, x] == 0 and [y, x] != self.agent_pos:
                return x, y

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.agent_pos = [0, 0]
        self.agent_health = 10
        self.agent_damage = 2
        self.has_sword = False
        
        x, y = self._get_random_free_call()
        self.grid[y, x] = 1
        self.sword_pos = [y, x]

        self.monsters = []
        for _ in range(3):
            x, y = self._get_random_free_call()
            self.grid[y, x] = 2
            self.monsters.append({"pos": [y, x], "health": 5})

        self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
        return self.grid.copy()
    
    def step(self, action):
        reward = 0
        done = False
        y, x = self.agent_pos

        if action == 0 and y > 0: self.agent_pos[0] -= 1
        elif action == 1 and y < self.grid_size - 1: self.agent_pos[0] += 1
        elif action == 2 and x > 0: self.agent_pos[1] -= 1
        elif action == 3 and x < self.grid_size - 1: self.agent_pos[1] += 1
        elif action == 4:
            attacked = False
            for monster in self.monsters:
                my, mx = monster["pos"]
                ay, ax = self.agent_pos
                if abs(my - ay) + abs(mx - ax) == 1:
                    dmg = 5 if self.has_sword else 2
                    monster["health"] -= dmg
                    attacked = True
                    print(f"👊 Agent attacks the monster ({mx}, {my}), damage: {dmg}")
                    if monster["health"] <= 0:
                        print(f"💀 Monster defeated at ({mx}, {my})!")
                        self.grid[my, mx] = 0
            self.monsters = [m for m in self.monsters if m["health"] > 0]
            if attacked:
                reward += 2

        elif action == 5:
            if self.agent_pos == self.sword_pos:
                self.has_sword = True
                self.sword_pos = [-1, -1]
                print("🗡️  Sword picked up!")
                reward += 1

        elif action == 6:
            reward += 0.2

        for m in self.monsters:
            my, mx = m["pos"]
            ay, ax = self.agent_pos
            if abs(my - ay) + abs(mx - ax) == 1:
                self.agent_health -= 1
                print(f"👾 Monster attacks! Agent loses 1 HP ({self.agent_health} left).")
                reward -= 1

        # Обновление сетки
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for m in self.monsters:
            y, x = m["pos"]
            self.grid[y, x] = 2
        if self.sword_pos != [-1, -1]:
            y, x = self.sword_pos
            self.grid[y, x] = 1
        ay, ax = self.agent_pos
        self.grid[ay, ax] = 3

        # Проверка конца игры
        if self.agent_health <= 0:
            print("☠️  Agent died!")
            reward -= 10
            done = True
        elif len(self.monsters) == 0:
            print("🏆 All the monsters are defeated!")
            reward += 10
            done = True

        return self.grid.copy(), reward, done, {}

    def render(self):
        if self.render_mode != "pygame":
            print(self.grid)
            return
        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.Font(None, 24)
            if self.agent_img is None:
                self.agent_img = pygame.image.load("assets/knight.png").convert_alpha()
                self.sword_img = pygame.image.load("assets/sword.png").convert_alpha()
                self.monster_img = pygame.image.load("assets/monster.png").convert_alpha()

        self.window.fill((30, 30, 30))

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (50, 50, 50), rect, 1)

        # Sword
        if self.sword_pos != [-1, -1]:
            sy, sx = self.sword_pos
            pygame.draw.circle(self.window, (255, 215, 0), (sx * self.cell_size + 50, sy * self.cell_size + 50), 15)

        # Monsters
        for m in self.monsters:
            y, x = m["pos"]
            pygame.draw.circle(self.window, (255, 0, 0), (x * self.cell_size + 50, y * self.cell_size + 50), 20)
            hp_text = self.font.render(str(m["health"]), True, (255, 255, 255))
            self.window.blit(hp_text, (x * self.cell_size + 5, y * self.cell_size + 5))

        # Agent
        ay, ax = self.agent_pos
        pygame.draw.circle(self.window, (0, 191, 255), (ax * self.cell_size + 50, ay * self.cell_size + 50), 20)
        hp_text = self.font.render(f"HP: {self.agent_health}", True, (0, 255, 0))
        self.window.blit(hp_text, (ax * self.cell_size + 5, ay * self.cell_size + 60))

        pygame.display.flip()
        pygame.event.pump()


env = MonsterHuntEnv(render_mode="pygame")
obs = env.reset()
done = False

while not done:
    env.render()
    action = None

    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP: action = 0
            elif event.key == pygame.K_DOWN: action = 1
            elif event.key == pygame.K_LEFT: action = 2
            elif event.key == pygame.K_RIGHT: action = 3
            elif event.key == pygame.K_SPACE: action = 4
            elif event.key == pygame.K_e: action = 5
            elif event.key == pygame.K_q: action = 6

    if action is not None:
        obs, reward, done, info = env.step(action)
        time.sleep(0.1)

pygame.quit()
