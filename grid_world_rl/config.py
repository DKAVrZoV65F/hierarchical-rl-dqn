# grid_world_rl/config.py

# Параметры среды
GRID_SIZE = 5
MAX_STEPS_PER_EPISODE = 100

# Параметры генерации среды
NUM_PUDDLES = 2              # Количество луж
NUM_NORMAL_WALLS = 2         # Количество обычных стен
NUM_JUMPABLE_WALLS = 2       # Количество перепрыгиваемых стен
NUM_CRAWLABLE_WALLS = 2      # Количество пролазных стен

# Параметры агента
STATE_SIZE = 2
ACTION_SIZE = 12
EPISODES = 50000  # Количество эпизодов обучения
BATCH_SIZE = 128
MODEL_FILENAME = 'dqn_agent_model'

# Параметры обучения
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999  # Медленное уменьшение ε
LEARNING_RATE = 0.0001

# Параметры сохранения моделей
MODEL_SAVE_FREQUENCY = EPISODES / 10  # Частота сохранения моделей (каждые N эпизодов)

# Параметры логирования
LOG_FILENAME = 'models/training_log.txt'  # Имя файла для сохранения логов

# Параметры визуализации
VISUALIZE_TRAINING = True  # Визуализировать обучение
VISUALIZE_TRAINING_FREQUENCY = EPISODES / 10  # Каждые N эпизодов
SAVE_ANIMATIONS = True  # Сохранять ли анимации в файл
ANIMATION_FORMAT = 'gif'  # Формат анимации: 'gif' или 'mp4'

# Отображение символов
SYMBOLS_MAPPING = {
    'puddle': 'P',          # Лужа
    'normal_wall': 'W',     # Обычная стена
    'jumpable_wall': 'J',   # Перепрыгиваемая стена
    'crawlable_wall': 'C',  # Пролазная стена
    'goal': 'G',            # Цель
    'agent': 'A',           # Агент
}

# Параметры HRL
HIGH_LEVEL_STATE_SIZE = 2          # Высокоуровневый агент использует то же состояние
HIGH_LEVEL_ACTION_SIZE = 2         # Количество подзадач
HIGH_LEVEL_EPISODES = EPISODES     # Количество эпизодов обучения высокоуровневого агента
LOW_LEVEL_STATE_SIZE = 2           # Низкоуровневые агенты используют то же состояние
LOW_LEVEL_ACTION_SIZE = 12         # Количество базовых действий