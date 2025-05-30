import pygame
from monster_hunt_env import MonsterHuntEnv
import time

# Инициализируем Pygame
pygame.init()

# Создаём окружение
env = MonsterHuntEnv(render_mode="pygame")
obs, info = env.reset()

# Убедимся, что Pygame окно создано
env.window = pygame.display.set_mode((env.window_size, env.window_size))
pygame.display.set_caption("Monster Hunt")
env.font = pygame.font.Font(None, 24)

# Только теперь можно загружать и конвертировать изображения
env.agent_img = pygame.image.load("assets/knight2.png").convert_alpha()
env.agent_img = pygame.transform.scale(env.agent_img, (env.cell_size, env.cell_size))

env.sword_img = pygame.image.load("assets/sword.png").convert_alpha()
env.sword_img = pygame.transform.scale(env.sword_img, (env.cell_size // 2, env.cell_size // 2))

env.monster_img = pygame.image.load("assets/monster3.png").convert_alpha()
env.monster_img = pygame.transform.scale(env.monster_img, (env.cell_size, env.cell_size))

# Основной цикл
running = True
clock = pygame.time.Clock()

# print("Управление: WASD — движение, J — атака, K — подобрать меч, L — ждать")

# while running:
#     env.render()
#
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#
#     keys = pygame.key.get_pressed()
#     action = None
#
#     if keys[pygame.K_w]:
#         action = 0
#     elif keys[pygame.K_s]:
#         action = 1
#     elif keys[pygame.K_a]:
#         action = 2
#     elif keys[pygame.K_d]:
#         action = 3
#     elif keys[pygame.K_j]:
#         action = 4
#     elif keys[pygame.K_k]:
#         action = 5
#     elif keys[pygame.K_l]:
#         action = 6
#
#     if action is not None:
#         obs, reward, terminated, truncated, info = env.step(action)
#         print(f"Action: {action}, Reward: {reward}")
#
#         if terminated:
#             print("Game Over!")
#             time.sleep(3)
#             obs, info = env.reset()
#
#     clock.tick(10)
#
# pygame.quit()
