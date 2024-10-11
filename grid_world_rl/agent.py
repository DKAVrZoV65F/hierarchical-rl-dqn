# grid_world_rl/agent.py

import numpy as np
from collections import deque
import random
import json
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from .config import GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size     # Размерность состояния
        self.action_size = action_size   # Количество действий
        self.memory = deque(maxlen=10000) # Память для хранения опыта

        self.gamma = GAMMA               # Дисконтирующий фактор
        self.epsilon = EPSILON           # Вероятность исследования
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE

        self.model = self._build_model()

    def _build_model(self):
        # Создание нейронной сети
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Сохранение опыта в памяти
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Недостаточно опыта для обучения

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        # Предсказания для текущих состояний и следующих состояний
        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)

        # Обновление целевых значений
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        # Обучение модели на всём батче данных
        self.model.fit(states, target, epochs=1, verbose=0)

    def update_epsilon(self):
        # Обновление ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def save(self, name):
        # Сохраняем модель
        self.model.save(f"{name}.keras")

        # Сохраняем параметры агента, такие как epsilon
        agent_params = {
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay
            # Добавьте другие параметры, если необходимо
        }
        with open(f"{name}_params.json", 'w') as f:
            json.dump(agent_params, f)

    def load(self, name):
        from tensorflow.keras.models import load_model
        # Загружаем модель
        self.model = load_model(f"{name}.keras", compile=False)
        # Компилируем модель заново
        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))

        # Загружаем параметры агента
        try:
            with open(f"{name}_params.json", 'r') as f:
                agent_params = json.load(f)
                self.epsilon = agent_params.get('epsilon', self.epsilon)
                self.epsilon_min = agent_params.get('epsilon_min', self.epsilon_min)
                self.epsilon_decay = agent_params.get('epsilon_decay', self.epsilon_decay)
                # Загрузите другие параметры, если они были сохранены
        except FileNotFoundError:
            print(f"Файл параметров агента {name}_params.json не найден. Использую параметры по умолчанию.")