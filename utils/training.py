# utils/training.py

import os
import numpy as np
import matplotlib.pyplot as plt
import logging  # Добавлено для логирования
from grid_world_rl.environment import GridWorld
from grid_world_rl.config import (
    GRID_SIZE,
    MAX_STEPS_PER_EPISODE,
    HIGH_LEVEL_EPISODES,
    BATCH_SIZE,
    VISUALIZE_TRAINING,
    VISUALIZE_TRAINING_FREQUENCY,
    SAVE_ANIMATIONS,
    MODEL_SAVE_FREQUENCY,  # Добавлено
)
from utils.visualization import animate_episode

def train_high_level_agent(high_level_agent, low_level_agent, high_level_model_path, low_level_model_path):
    # Списки для хранения статистики
    episode_rewards = []
    episode_steps = []
    epsilon_values = []
    episode_jumps = []
    episode_crawls = []
    episode_puddle_hits = []
    episode_wall_collisions = []
    successful_episodes = 0  # Счётчик успешных эпизодов
    best_success_rate = 0    # Лучший показатель успешности

    print('Начало обучения агентов...')
    logging.info('Начало обучения агентов...')

    for e in range(HIGH_LEVEL_EPISODES):
        env = GridWorld(GRID_SIZE)
        state = env.reset()
        state = np.array(state)
        done = False

        time_step = 0
        total_episode_reward = 0
        total_episode_steps = 0

        path = [state.tolist()]  # Для визуализации пути

        # Инициализация счётчиков
        action_counts = {'move': 0, 'jump': 0, 'crawl': 0}
        jumps = 0
        crawls = 0
        puddle_hits = 0
        wall_collisions = 0

        while not done and time_step < MAX_STEPS_PER_EPISODE:
            # Высокоуровневый агент выбирает опцию
            option = high_level_agent.choose_option(state)

            # Логика выбора подзадачи
            if option == 0:
                # Опция 0: навигация к цели
                subgoal = env.goal
                sub_done = False
            elif option == 1:
                # Опция 1: преодоление препятствий
                subgoal = None
                sub_done = False

            # Выполнение подзадачи низкоуровневым агентом
            sub_step = 0
            total_sub_reward = 0

            while not sub_done:
                action = low_level_agent.choose_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.array(next_state)

                time_step += 1
                sub_step += 1
                total_episode_steps += 1
                total_sub_reward += reward
                total_episode_reward += reward

                # Обновление счётчиков действий
                if action in [0, 1, 2, 3]:
                    action_counts['move'] += 1
                elif action in [4, 5, 6, 7]:
                    action_counts['jump'] += 1
                    jumps += 1
                elif action in [8, 9, 10, 11]:
                    action_counts['crawl'] += 1
                    crawls += 1

                # Сбор событий из info
                if info.get('puddle_hit', False):
                    puddle_hits += 1
                if info.get('wall_collision', False) or info.get('jumpable_wall_hit', False) or info.get('crawlable_wall_hit', False):
                    wall_collisions += 1

                path.append(next_state.tolist())

                # Низкоуровневый агент запоминает переход
                low_level_agent.remember(state, action, reward, next_state, done)

                state = next_state

                # Условия завершения подзадачи
                if option == 0:
                    if all(state == env.goal) or sub_step >= MAX_STEPS_PER_EPISODE:
                        sub_done = True
                elif option == 1:
                    sub_done = True

                if done or time_step >= MAX_STEPS_PER_EPISODE:
                    sub_done = True

                # Обучение низкоуровневого агента
                low_level_agent.replay(BATCH_SIZE)

            # Высокоуровневый агент запоминает переход
            high_level_agent.remember(state, option, total_sub_reward, state, done)

            # Обучение высокоуровневого агента
            high_level_agent.replay(BATCH_SIZE)

            if done or time_step >= MAX_STEPS_PER_EPISODE:
                if all(state == env.goal):
                    successful_episodes += 1
                current_success_rate = successful_episodes / (e + 1)
                print(f"Episode: {e+1}/{HIGH_LEVEL_EPISODES}, Steps: {total_episode_steps}, Reward: {total_episode_reward:.2f}, Epsilon: {high_level_agent.agent.epsilon:.2f}, Success Rate: {current_success_rate:.2f}")
                print(f"Actions taken: Move: {action_counts['move']}, Jump: {action_counts['jump']}, Crawl: {action_counts['crawl']}")
                print(f"Jumps: {jumps}, Crawls: {crawls}, Puddle Hits: {puddle_hits}, Wall Collisions: {wall_collisions}")

                # Логирование результатов
                logging.info(f"Episode: {e+1}/{HIGH_LEVEL_EPISODES}, Steps: {total_episode_steps}, Reward: {total_episode_reward:.2f}, Epsilon: {high_level_agent.agent.epsilon:.2f}, Success Rate: {current_success_rate:.2f}")
                logging.info(f"Actions taken: Move: {action_counts['move']}, Jump: {action_counts['jump']}, Crawl: {action_counts['crawl']}")
                logging.info(f"Jumps: {jumps}, Crawls: {crawls}, Puddle Hits: {puddle_hits}, Wall Collisions: {wall_collisions}")

                # Собираем статистику
                episode_rewards.append(total_episode_reward)
                episode_steps.append(total_episode_steps)
                epsilon_values.append(high_level_agent.agent.epsilon)
                episode_jumps.append(jumps)
                episode_crawls.append(crawls)
                episode_puddle_hits.append(puddle_hits)
                episode_wall_collisions.append(wall_collisions)

                # Проверяем, улучшилась ли успешность
                if current_success_rate > best_success_rate:
                    best_success_rate = current_success_rate
                    # Сохраняем модели с наилучшей успешностью
                    high_level_agent.save(high_level_model_path + '_best')
                    low_level_agent.save(low_level_model_path + '_best')
                    print('Найдена новая лучшая модель, модели сохранены.')
                    logging.info('Найдена новая лучшая модель, модели сохранены.')

                break

        # Обновление ε после каждого эпизода
        high_level_agent.update_epsilon()
        low_level_agent.update_epsilon()

        # Периодическое сохранение моделей
        if (e + 1) % MODEL_SAVE_FREQUENCY == 0:
            high_level_agent.save(high_level_model_path)
            low_level_agent.save(low_level_model_path)
            print(f'Модели сохранены после {e + 1} эпизодов.')
            logging.info(f'Модели сохранены после {e + 1} эпизодов.')

        # Визуализация обучения каждые N эпизодов
        if VISUALIZE_TRAINING and (e + 1) % VISUALIZE_TRAINING_FREQUENCY == 0:
            title = f"Обучение: Эпизод {e + 1}"
            filename = None
            if SAVE_ANIMATIONS:
                filename = f"animations/training_episode_{e + 1}"
            animate_episode(GRID_SIZE, env.puddles, env.walls, path, env.goal, title, filename)

    # После завершения обучения сохраняем последнюю модель
    os.makedirs('models', exist_ok=True)
    high_level_agent.save(high_level_model_path)
    low_level_agent.save(low_level_model_path)
    print('Модели агентов сохранены.')
    logging.info('Модели агентов сохранены.')

    print(f"Количество успешно достигнутых целей за обучение: {successful_episodes}/{HIGH_LEVEL_EPISODES}")
    logging.info(f"Количество успешно достигнутых целей за обучение: {successful_episodes}/{HIGH_LEVEL_EPISODES}")

    # Построение графиков обучения (опционально)
    plot_training_statistics(
        episode_rewards,
        episode_steps,
        epsilon_values,
        episode_jumps,
        episode_crawls,
        episode_puddle_hits,
        episode_wall_collisions
    )

    # Построение графиков обучения
    plot_training_statistics(
        episode_rewards,
        episode_steps,
        epsilon_values,
        episode_jumps,
        episode_crawls,
        episode_puddle_hits,
        episode_wall_collisions
    )

def plot_training_statistics(episode_rewards, episode_steps, epsilon_values, episode_jumps, episode_crawls, episode_puddle_hits, episode_wall_collisions):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(2, 2, 2)
    plt.plot(episode_steps)
    plt.title('Total Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Steps')

    plt.subplot(2, 2, 3)
    plt.plot(epsilon_values)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.subplot(2, 2, 4)
    plt.plot(episode_jumps, label='Jumps')
    plt.plot(episode_crawls, label='Crawls')
    plt.plot(episode_puddle_hits, label='Puddle Hits')
    plt.plot(episode_wall_collisions, label='Wall Collisions')
    plt.title('Actions and Events per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/hrl_training_statistics.png')
    plt.show()