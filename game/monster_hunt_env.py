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
        self.observation_space = spaces.Box(low=0, high=3, shape=(28,), dtype=np.float32)

        self.agent_img = None
        self.sword_img = None
        self.monster_img = None

    def _get_random_free_call(self):
        while True:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if self.grid[y, x] == 0 and [y, x] != self.agent_pos:
                return x, y

    def _get_obs(self):
        base_grid = self.grid.flatten()
        additional_info = np.array([
            self.agent_health / 10.0,
            1.0 if self.has_sword else 0.0,
            *[m["health"] / 5.0 for m in self.monsters]
        ])
        return np.concatenate([base_grid, additional_info]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            options = {}

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.agent_pos = [0, 0]
        self.agent_health = 10
        self.agent_damage = 2
        self.has_sword = False

        x, y = self._get_random_free_call()
        self.grid[2, 2] = 1
        self.sword_pos = [2, 2]

        self.monsters = []
        for _ in range(2):
            x, y = self._get_random_free_call()
            self.grid[y, x] = 2
            self.monsters.append({"pos": [y, x], "health": 5})

        self.grid[self.agent_pos[0], self.agent_pos[1]] = 3
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        reward = 0
        done = False
        terminated = False
        truncated = False
        info = {}
        prev_distance_to_sword = abs(self.agent_pos[0] - self.sword_pos[0]) + abs(self.agent_pos[1] - self.sword_pos[1])

        y, x = self.agent_pos

        if action in [0, 1, 2, 3]:
            new_y, new_x = self.agent_pos
            if action == 0 and new_y > 0:
                new_y -= 1
            elif action == 1 and new_y < self.grid_size - 1:
                new_y += 1
            elif action == 2 and new_x > 0:
                new_x -= 1
            elif action == 3 and new_x < self.grid_size - 1:
                new_x += 1

            # Штраф за шаг, если не приближается к мечу или монстрам
            new_distance_to_sword = abs(new_y - self.sword_pos[0]) + abs(new_x - self.sword_pos[1])
            if new_distance_to_sword >= prev_distance_to_sword:
                reward -= 1

            self.agent_pos = [new_y, new_x]

        elif action == 4:
            attacked = False
            for monster in self.monsters:
                my, mx = monster["pos"]
                ay, ax = self.agent_pos
                if abs(my - ay) + abs(mx - ax) == 1:
                    dmg = 5 if self.has_sword else 2
                    monster["health"] -= dmg
                    attacked = True
                    print(f"Agent attacks the monster ({mx}, {my}), damage: {dmg}")
                    if monster["health"] <= 0:
                        print(f"Monster defeated at ({mx}, {my})!")
                        self.grid[my, mx] = 0
            self.monsters = [m for m in self.monsters if m["health"] > 0]
            if attacked:
                if self.has_sword:
                    reward += 200
                else:
                    reward -= 50

        elif action == 5:
            if self.agent_pos == self.sword_pos and not self.has_sword:
                reward += 300
                self.has_sword = True
                self.sword_pos = [-1, -1]
                print("Sword picked up!")

        elif action == 6:
            reward -= 5

        for m in self.monsters:
            my, mx = m["pos"]
            ay, ax = self.agent_pos
            if abs(my - ay) + abs(mx - ax) == 1:
                self.agent_health -= 1
                print(f"Monster attacks! Agent loses 1 HP ({self.agent_health} left).")
                reward -= 50

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
            print("Agent died!")
            reward -= 100
            done = True
            terminated = True
        elif len(self.monsters) == 0:
            print("All the monsters are defeated!")
            reward += 200
            done = True
            terminated = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "pygame":
            print(self.grid)
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.Font(None, 24)
            if self.agent_img is None:
                self.agent_img = pygame.image.load("assets/knight2.png").convert_alpha()
                self.sword_img = pygame.image.load("assets/sword.png").convert_alpha()
                self.monster_img = pygame.image.load("assets/monster3.png").convert_alpha()

        self.window.fill((30, 30, 30))

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (50, 50, 50), rect, 1)

        # Sword
        if self.sword_pos != [-1, -1]:
            sy, sx = self.sword_pos
            self.window.blit(self.sword_img, (sx * self.cell_size, sy * self.cell_size))

        # Monsters
        for m in self.monsters:
            y, x = m["pos"]
            self.window.blit(self.monster_img, (x * self.cell_size, y * self.cell_size))
            hp_text = self.font.render(str(m["health"]), True, (255, 255, 255))
            self.window.blit(hp_text, (x * self.cell_size + 5, y * self.cell_size + 5))

        # Agent
        ay, ax = self.agent_pos
        self.window.blit(self.agent_img, (ax * self.cell_size, ay * self.cell_size))
        hp_text = self.font.render(f"HP: {self.agent_health}", True, (0, 255, 0))
        self.window.blit(hp_text, (ax * self.cell_size + 5, ay * self.cell_size + 60))

        pygame.display.flip()
        pygame.event.pump()
