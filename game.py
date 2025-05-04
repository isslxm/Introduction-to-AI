import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class MonsterHuntEnv(gym.Env):
    metadata = {'render.modes': ['human'], "render_fps":2}

    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = 5
        self.max_monsters = 3
        self.max_steps = 100

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=4, shape= (5, 5), dtype=np.uint8),
            "has_sword": spaces.Discrete(2)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
 
        self.agent_pos = [0, 0]
        self.has_sword = False
        self.steps = 0

        self.grid = np.zeros((5, 5), dtype=np.uint8)
        
        sword_pos = [random.randint(0, 4), random.randint(0, 4)]
        self.grid[sword_pos] = 1

        self.monsters = []
        while len(self.monsters) < self.max_monsters:
            pos = (random.randint(0, 4), random.randint(0, 4))
            if self.grid[pos] == 0 and pos != tuple(self.agent_pos):
                self.grid[pos] = 2
                self.monsters.append(pos)

        return self._get_obs(), {}
    
    def _get_obs(self):
        return {
            "grid": self.grid.copy(),
            "has_sword": int(self.has_sword)
        }
    
    def step(self, action):
        self.steps += 1
        reward = -0.1
        done = False

        y, x = self.agent_pos

        def in_bounds(ny, nx):
            return 0 <= ny < self.grid_size and 0 <= nx < self.grid_size
        
        if action == 0 and in_bounds(y - 1, x):  # вверх
            self.agent_pos[0] -= 1
        elif action == 1 and in_bounds(y + 1, x):  # вниз
            self.agent_pos[0] += 1
        elif action == 2 and in_bounds(y, x - 1):  # влево
            self.agent_pos[1] -= 1
        elif action == 3 and in_bounds(y, x + 1):  # вправо
            self.agent_pos[1] += 1
        elif action == 4:  # атака
            if not self.has_sword:
                reward -= 1  # нельзя атаковать без меча
            else:
                killed = False
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if in_bounds(ny, nx) and self.grid[ny, nx] == 2:
                        self.grid[ny, nx] = 0
                        self.monsters.remove((ny, nx))
                        reward += 5
                        killed = True
                        break
                if not killed:
                    reward -= 0.5  # атака впустую
        elif action == 5:  # взять предмет
            if self.grid[y, x] == 1:
                self.has_sword = True
                self.grid[y, x] = 0
                reward += 1
        elif action == 6:  # блокировать
            reward += 0.2  # возможно, пригодится в будущих версиях

        # Конец игры
        if len(self.monsters) == 0:
            done = True
            reward += 10

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self):
        display = self.grid.copy()
        y, x = self.agent_pos
        display[y, x] = 4  # отобразим агента
        print(display)
        print(f"Sword: {self.has_sword}, Monsters left: {len(self.monsters)}")

    def close(self):
        pass

env = MonsterHuntEnv()

obs, _ = env.reset()
done = False

while not done:
    env.render()
    action = int(input("Action (0-up, 1-down, 2-left, 3-right, 4-hit, 5-pick up, 6-block): "))
    obs, reward, done, _, _ = env.step(action)
    print(f"Reward: {reward}")
