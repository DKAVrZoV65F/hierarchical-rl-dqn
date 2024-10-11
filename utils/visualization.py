# utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import warnings
from grid_world_rl.config import (
    SAVE_ANIMATIONS,
    ANIMATION_FORMAT,
    SYMBOLS_MAPPING,
)

def animate_episode(grid_size, puddles, walls, path, goal, title, filename=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.close()  # Закрываем фигуру, чтобы не отображалась во время создания
    fig.subplots_adjust(right=0.8)

    # Создаем пустую сетку
    grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]

    # Размещаем элементы на сетке согласно SYMBOLS_MAPPING
    for puddle in puddles:
        x, y = puddle
        grid[x][y] = SYMBOLS_MAPPING.get('puddle', 'P')
    for wall_type, positions in walls.items():
        symbol = SYMBOLS_MAPPING.get(wall_type, wall_type[0].upper())
        for wall in positions:
            x, y = wall
            grid[x][y] = symbol
    gx, gy = goal
    grid[gx][gy] = SYMBOLS_MAPPING.get('goal', 'G')

    # Функция для обновления кадра анимации
    def update(frame):
        ax.clear()
        current_grid = [row[:] for row in grid]  # Копируем оригинальную сетку
        # Размещаем агента на сетке
        x, y = path[frame]
        if current_grid[x][y] not in (SYMBOLS_MAPPING.get('goal', 'G'),):
            current_grid[x][y] = SYMBOLS_MAPPING.get('agent', 'A')
        # Отображение сетки
        ax.imshow([[1]*grid_size]*grid_size, cmap='Greys', extent=(0, grid_size, 0, grid_size))

        # Отображение элементов на карте
        for i in range(grid_size):
            for j in range(grid_size):
                cell = current_grid[i][j]
                text = cell
                color = 'black'
                if cell == SYMBOLS_MAPPING.get('puddle', 'P'):
                    color = 'blue'
                elif cell == SYMBOLS_MAPPING.get('normal_wall', 'W'):
                    color = 'black'
                elif cell == SYMBOLS_MAPPING.get('jumpable_wall', 'J'):
                    color = 'orange'
                elif cell == SYMBOLS_MAPPING.get('crawlable_wall', 'C'):
                    color = 'brown'
                elif cell == SYMBOLS_MAPPING.get('goal', 'G'):
                    color = 'green'
                elif cell == SYMBOLS_MAPPING.get('agent', 'A'):
                    color = 'red'
                ax.text(j + 0.5, grid_size - i - 0.5, text, ha='center', va='center', color=color, fontsize=12)
        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)

        # Легенда
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Агент', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Лужа', markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Обычная стена', markerfacecolor='black', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Перепрыгиваемая стена', markerfacecolor='orange', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Пролазная стена', markerfacecolor='brown', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Цель', markerfacecolor='green', markersize=10),
        ]
        # Перемещаем легенду в левый верхний угол рядом с картой
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    anim = animation.FuncAnimation(fig, update, frames=len(path), interval=500, repeat=False)

    # Увеличиваем отступы, чтобы не обрезалось изображение
    plt.tight_layout()

    if SAVE_ANIMATIONS and filename:
        if ANIMATION_FORMAT == 'gif':
            anim.save(f"{filename}.gif", writer='pillow')
        elif ANIMATION_FORMAT == 'mp4':
            anim.save(f"{filename}.mp4", writer='ffmpeg')
        else:
            warnings.warn("Неподдерживаемый формат анимации. Доступны 'gif' и 'mp4'.")
    else:
        plt.show()