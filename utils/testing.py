# utils/testing.py

import os
import numpy as np
import matplotlib.pyplot as plt
from grid_world_rl.environment import GridWorld
from grid_world_rl.config import (
    GRID_SIZE,
    MAX_STEPS_PER_EPISODE,
    SAVE_ANIMATIONS,
)
from utils.visualization import animate_episode

def test_high_level_agent(high_level_agent, low_level_agent):
    test_successful_episodes = 0

    # Устанавливаем ε в 0 для тестирования
    high_level_agent.agent.epsilon = 0.0
    low_level_agent.agent.epsilon = 0.0

    print('Тестирование агентов на новых картах...')

    # Списки для хранения статистики по эпизодам
    test_episode_rewards = []
    test_episode_steps = []
    test_episode_jumps = []
    test_episode_crawls = []
    test_episode_puddle_hits = []
    test_episode_wall_collisions = []

    for i in range(5):
        env = GridWorld(GRID_SIZE)
        state = env.reset()
        state = np.array(state)
        done = False
        time_step = 0
        path = [state.tolist()]

        # Инициализация статистики для текущего эпизода
        total_episode_reward = 0
        total_episode_steps = 0
        jumps = 0
        crawls = 0
        puddle_hits = 0
        wall_collisions = 0

        while not done:
            # Высокоуровневый агент выбирает опцию
            option = high_level_agent.choose_option(state)

            if option == 0:
                subgoal = env.goal
                sub_done = False
            elif option == 1:
                subgoal = None
                sub_done = False

            # Выполнение подзадачи низкоуровневым агентом
            sub_step = 0

            while not sub_done:
                action = low_level_agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.array(next_state)

                time_step += 1
                sub_step += 1
                total_episode_steps += 1
                total_episode_reward += reward

                # Обновление счётчиков действий
                if action in [4, 5, 6, 7]:
                    jumps += 1
                elif action in [8, 9, 10, 11]:
                    crawls +=1

                # Сбор событий из info
                if info.get('puddle_hit', False):
                    puddle_hits += 1
                if info.get('wall_collision', False) or info.get('jumpable_wall_hit', False) or info.get('crawlable_wall_hit', False):
                    wall_collisions += 1

                path.append(state.tolist())
                state = next_state

                # Условия завершения подзадачи
                if option == 0:
                    if all(state == env.goal) or sub_step >= MAX_STEPS_PER_EPISODE:
                        sub_done = True
                elif option == 1:
                    sub_done = True

                if done or time_step >= MAX_STEPS_PER_EPISODE:
                    sub_done = True
                    done = True

            if done or time_step >= MAX_STEPS_PER_EPISODE:
                break

        # После завершения тестового эпизода
        test_episode_rewards.append(total_episode_reward)
        test_episode_steps.append(total_episode_steps)
        test_episode_jumps.append(jumps)
        test_episode_crawls.append(crawls)
        test_episode_puddle_hits.append(puddle_hits)
        test_episode_wall_collisions.append(wall_collisions)

        # Выводим результаты
        print(f"Карта {i+1}:")
        print(f"Количество шагов: {total_episode_steps}, Общая награда: {total_episode_reward}")
        print(f"Количество прыжков: {jumps}")
        print(f"Количество ползаний: {crawls}")
        print(f"Попаданий в лужу: {puddle_hits}")
        print(f"Столкновений со стеной: {wall_collisions}")

        if all(state == env.goal):
            print("Агент достиг цели!")
            test_successful_episodes += 1
        else:
            print("Агент не смог достичь цели.")

        print("Путь агента:")

        # Визуализация
        title = f"Тестирование: Карта {i+1}"
        filename = None
        if SAVE_ANIMATIONS:
            filename = f"animations/test_episode_{i + 1}"
        animate_episode(GRID_SIZE, env.puddles, env.walls, path, env.goal, title, filename)

    print(f"Количество успешно достигнутых целей при тестировании: {test_successful_episodes}/5")

    # Построение графиков статистики
    plot_testing_statistics(
        test_episode_rewards,
        test_episode_steps,
        test_episode_jumps,
        test_episode_crawls,
        test_episode_puddle_hits,
        test_episode_wall_collisions
    )

def plot_testing_statistics(test_episode_rewards, test_episode_steps, test_episode_jumps, test_episode_crawls, test_episode_puddle_hits, test_episode_wall_collisions):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(test_episode_rewards)
    plt.title('Total Reward per Test Episode')
    plt.xlabel('Test Episode')
    plt.ylabel('Total Reward')

    plt.subplot(2, 2, 2)
    plt.plot(test_episode_steps)
    plt.title('Total Steps per Test Episode')
    plt.xlabel('Test Episode')
    plt.ylabel('Total Steps')

    plt.subplot(2, 2, 3)
    plt.plot(test_episode_jumps, label='Jumps')
    plt.plot(test_episode_crawls, label='Crawls')
    plt.plot(test_episode_puddle_hits, label='Puddle Hits')
    plt.plot(test_episode_wall_collisions, label='Wall Collisions')
    plt.title('Actions and Events per Test Episode')
    plt.xlabel('Test Episode')
    plt.ylabel('Count')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/hrl_testing_statistics.png')
    plt.show()