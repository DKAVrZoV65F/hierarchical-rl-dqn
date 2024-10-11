# grid_world_rl/environment.py
from collections import deque

from .config import NUM_PUDDLES, NUM_NORMAL_WALLS, NUM_JUMPABLE_WALLS, NUM_CRAWLABLE_WALLS, MAX_STEPS_PER_EPISODE
import random

class GridWorld:
    def __init__(self, grid_size):
        self.grid_size = grid_size  # Размер сетки
        self.puddles = []           # Позиции луж
        self.walls = {
            'normal': [],
            'jumpable': [],
            'crawlable': []
        }
        self.goal = None
        self.position = (0, 0)
        self.steps_taken = 0        # Счетчик шагов
        self.reset()

    def reset(self):
        self.position = (0, 0)
        self.generate_random_environment()
        self.steps_taken = 0        # Сбрасываем счетчик шагов при сбросе среды
        return self.position

    def is_inside_grid(self, position):
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def is_valid(self, position):
        if not self.is_inside_grid(position):
            return False
        if position in self.puddles or position in self.walls['normal'] \
           or position in self.walls['jumpable'] or position in self.walls['crawlable']:
            return False
        return True

    def can_jump_over(self, over_pos):
        return over_pos in self.walls['jumpable'] or over_pos in self.puddles

    def can_crawl_through(self, position):
        return position in self.walls['crawlable']

    def generate_random_environment(self):
        # Генерация случайных позиций для луж, стен и цели
        positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                     if (x, y) != self.position]

        self.puddles = random.sample(positions, NUM_PUDDLES)
        positions = [pos for pos in positions if pos not in self.puddles]

        self.walls['normal'] = random.sample(positions, NUM_NORMAL_WALLS)
        positions = [pos for pos in positions if pos not in self.walls['normal']]

        self.walls['jumpable'] = random.sample(positions, NUM_JUMPABLE_WALLS)
        positions = [pos for pos in positions if pos not in self.walls['jumpable']]

        self.walls['crawlable'] = random.sample(positions, NUM_CRAWLABLE_WALLS)
        positions = [pos for pos in positions if pos not in self.walls['crawlable']]

        self.goal = random.choice(positions)

    def step(self, action):
        x, y = self.position
        reward = 0
        info = {'puddle_hit': False, 'wall_collision': False, 'jumpable_wall_hit': False, 'crawlable_wall_hit': False}
        done = False

        # Параметры функции вознаграждения
        collision_penalty = -5         # Штраф за столкновение со стеной
        puddle_penalty = -5            # Штраф за попадание в лужу
        step_cost = -1                 # Штраф за каждый шаг
        goal_reward = 200              # Базовое вознаграждение за достижение цели
        success_reward = 10            # Вознаграждение за успешный прыжок или ползание
        # Начисление бонуса за оставшиеся шаги при достижении цели
        remaining_steps_bonus = 0

        # Определяем следующее положение в зависимости от действия
        if action == 0:  # Up
            next_pos = (x - 1, y)
        elif action == 1:  # Down
            next_pos = (x + 1, y)
        elif action == 2:  # Left
            next_pos = (x, y - 1)
        elif action == 3:  # Right
            next_pos = (x, y + 1)
        elif action >= 4 and action <= 7:  # Jump actions
            if action == 4:  # Jump Up
                over_pos = (x - 1, y)
                next_pos = (x - 2, y)
            elif action == 5:  # Jump Down
                over_pos = (x + 1, y)
                next_pos = (x + 2, y)
            elif action == 6:  # Jump Left
                over_pos = (x, y - 1)
                next_pos = (x, y - 2)
            elif action == 7:  # Jump Right
                over_pos = (x, y + 1)
                next_pos = (x, y + 2)
            return self.jump(over_pos, next_pos, info, success_reward, collision_penalty, goal_reward)
        elif action >= 8 and action <= 11:  # Crawl actions
            if action == 8:  # Crawl Up
                next_pos = (x - 1, y)
            elif action == 9:  # Crawl Down
                next_pos = (x + 1, y)
            elif action == 10:  # Crawl Left
                next_pos = (x, y - 1)
            elif action == 11:  # Crawl Right
                next_pos = (x, y + 1)
            return self.crawl(next_pos, info, success_reward, collision_penalty, goal_reward)
        else:
            next_pos = (x, y)

        # Инкрементируем счетчик шагов
        self.steps_taken += 1

        # Штраф за каждый шаг
        reward += step_cost

        # Обработка движения
        if not self.is_inside_grid(next_pos):
            reward += collision_penalty
            info['wall_collision'] = True
        elif next_pos == self.goal:
            self.position = next_pos
            # Начисляем базовое вознаграждение за достижение цели
            reward += goal_reward
            # Добавляем бонус за оставшиеся шаги
            remaining_steps_bonus = MAX_STEPS_PER_EPISODE - self.steps_taken
            reward += remaining_steps_bonus
            done = True
        elif next_pos in self.walls['normal'] or next_pos in self.walls['jumpable'] or next_pos in self.walls['crawlable']:
            reward += collision_penalty
            info['wall_collision'] = True
        elif next_pos in self.puddles:
            self.position = next_pos
            reward += puddle_penalty
            info['puddle_hit'] = True
        else:
            self.position = next_pos

        return self.position, reward, done, info

    def jump(self, over_pos, next_pos, info, success_reward, collision_penalty, goal_reward):
        reward = -2  # Стоимость прыжка
        done = False

        # Инкрементируем счетчик шагов
        self.steps_taken += 1

        # Штраф за шаг
        reward += -1  # Штраф за шаг

        if not self.is_inside_grid(next_pos):
            reward += collision_penalty
            info['wall_collision'] = True
        elif self.can_jump_over(over_pos) and self.is_valid(next_pos):
            self.position = next_pos
            reward += success_reward  # Вознаграждение за успешный прыжок
            if self.position == self.goal:
                reward += goal_reward
                # Добавляем бонус за оставшиеся шаги
                remaining_steps_bonus = MAX_STEPS_PER_EPISODE - self.steps_taken
                reward += remaining_steps_bonus
                done = True
        else:
            reward += collision_penalty
            if over_pos in self.walls['normal']:
                info['wall_collision'] = True
            elif over_pos in self.walls['jumpable']:
                info['jumpable_wall_hit'] = True
            elif over_pos in self.puddles:
                info['puddle_hit'] = True

        return self.position, reward, done, info

    def crawl(self, next_pos, info, success_reward, collision_penalty, goal_reward):
        reward = -2  # Стоимость ползания
        done = False

        # Инкрементируем счетчик шагов
        self.steps_taken += 1

        # Штраф за шаг
        reward += -1  # Штраф за шаг

        if not self.is_inside_grid(next_pos):
            reward += collision_penalty
            info['wall_collision'] = True
        elif self.can_crawl_through(next_pos):
            self.position = next_pos
            reward += success_reward  # Вознаграждение за успешное ползание
            if self.position == self.goal:
                reward += goal_reward
                # Добавляем бонус за оставшиеся шаги
                remaining_steps_bonus = MAX_STEPS_PER_EPISODE - self.steps_taken
                reward += remaining_steps_bonus
                done = True
        else:
            reward += collision_penalty
            if next_pos in self.walls['crawlable']:
                info['crawlable_wall_hit'] = True
            elif next_pos in self.walls['normal'] or next_pos in self.walls['jumpable']:
                info['wall_collision'] = True

        return self.position, reward, done, info

    def is_inside_grid(self, position):
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def generate_random_environment(self):
        # Генерация случайных позиций для луж, стен и цели
        positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                     if (x, y) != self.position]

        self.puddles = random.sample(positions, NUM_PUDDLES)
        positions = [pos for pos in positions if pos not in self.puddles]

        self.walls['normal'] = random.sample(positions, NUM_NORMAL_WALLS)
        positions = [pos for pos in positions if pos not in self.walls['normal']]

        self.walls['jumpable'] = random.sample(positions, NUM_JUMPABLE_WALLS)
        positions = [pos for pos in positions if pos not in self.walls['jumpable']]

        self.walls['crawlable'] = random.sample(positions, NUM_CRAWLABLE_WALLS)
        positions = [pos for pos in positions if pos not in self.walls['crawlable']]

        self.goal = random.choice(positions)

    def is_goal_reachable(self):
        visited = set()
        queue = deque()
        queue.append(self.position)
        visited.add(self.position)

        while queue:
            current = queue.popleft()
            if current == self.goal:
                return True
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and self.is_valid_for_reachability(neighbor):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def is_valid_for_reachability(self, position):
        x, y = position
        # Проверка границ сетки
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        # Обычные стены непроходимы, но перепрыгиваемые и пролазные стены считаются проходимыми для оценки достижимости
        if position in self.walls['normal']:
            return False
        return True

    def get_neighbors(self, position):
        x, y = position
        neighbors = [
            (x - 1, y),  # Up
            (x + 1, y),  # Down
            (x, y - 1),  # Left
            (x, y + 1)   # Right
        ]
        # Фильтруем соседей, чтобы оставаться внутри границ сетки
        valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size]
        return valid_neighbors