# main.py

import os
import sys
from sys import platform

import tensorflow as tf
import logging
from grid_world_rl.hierarchical_agent import HighLevelAgent, LowLevelAgent
from grid_world_rl.config import (
    STATE_SIZE,
    ACTION_SIZE,
    HIGH_LEVEL_EPISODES,
    BATCH_SIZE,
    MODEL_FILENAME,
    LOG_FILENAME,
)
from utils.training import train_high_level_agent
from utils.testing import test_high_level_agent

if __name__ == "__main__":
    sys.tracebacklimit = None  # Не показывать полные трассировки исключений

    if sys.platform == 'darwin':
        # Удаляем или комментируем строки, отключающие GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        tf.config.set_visible_devices([], 'GPU')

    # Настройка логирования
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Проверяем доступность GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPUs: ", tf.config.list_physical_devices('GPU'))

    # Создаем экземпляры агентов
    high_level_agent = HighLevelAgent()
    low_level_agent = LowLevelAgent()
    high_level_model_path = os.path.join('models', 'high_level_' + MODEL_FILENAME)
    low_level_model_path = os.path.join('models', 'low_level_' + MODEL_FILENAME)

    # Проверяем, существуют ли сохраненные модели
    if os.path.exists(f'{high_level_model_path}.keras') and os.path.exists(f'{low_level_model_path}.keras'):
        # Загрузка моделей для продолжения обучения
        high_level_agent.load(high_level_model_path)
        low_level_agent.load(low_level_model_path)
        print('Загружены сохраненные модели агентов. Продолжаем обучение.')
        logging.info('Загружены сохраненные модели агентов. Продолжаем обучение.')
    else:
        print('Сохраненные модели не найдены. Начинаем новое обучение.')
        logging.info('Сохраненные модели не найдены. Начинаем новое обучение.')

    # Продолжаем обучение агентов
    train_high_level_agent(high_level_agent, low_level_agent, high_level_model_path, low_level_model_path)

    # Тестирование агентов на новых картах после завершения обучения
    test_high_level_agent(high_level_agent, low_level_agent)