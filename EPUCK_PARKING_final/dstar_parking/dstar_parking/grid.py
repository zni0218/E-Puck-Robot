"""
grid.py  --  A GRELHA DE OCUPAÇÃO QUE O D* Lite USA POR DENTRO
==============================================================
É uma matriz de células: 0 = livre, 255 = ocupada (parede). Sabe responder a duas
perguntas que o algoritmo faz o tempo todo:
  - "esta célula está livre?"  -> is_unoccupied
  - "quem são os vizinhos desta célula?"  -> succ  (8 vizinhos: cima/baixo/lados/diagonais)

NOTA: a célula é (x, y) = (linha, coluna). É a mesma convenção (gx, gy) da ParkingMap.
"""

import numpy as np
from utils import get_movements_4n, get_movements_8n

OBSTACLE = 255
UNOCCUPIED = 0


class OccupancyGridMap:
    def __init__(self, x_dim, y_dim, exploration_setting='8N'):
        # x_dim = nº de linhas, y_dim = nº de colunas. Matriz começa toda livre (0).
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.occupancy_grid_map = np.zeros((x_dim, y_dim), dtype=np.uint8)
        self.visited = {}
        self.exploration_setting = exploration_setting   # '4N' (sem diagonais) ou '8N'

    def is_unoccupied(self, pos):
        # True se a célula está LIVRE (== 0). round() garante índices inteiros.
        row, col = round(pos[0]), round(pos[1])
        return self.occupancy_grid_map[row][col] == UNOCCUPIED

    def in_bounds(self, cell):
        # A célula está dentro da matriz?
        x, y = cell
        return 0 <= x < self.x_dim and 0 <= y < self.y_dim

    def filter(self, neighbors, avoid_obstacles):
        # Mantém só os vizinhos dentro dos limites; se avoid_obstacles, também só os livres.
        if avoid_obstacles:
            return [n for n in neighbors if self.in_bounds(n) and self.is_unoccupied(n)]
        return [n for n in neighbors if self.in_bounds(n)]

    def succ(self, vertex, avoid_obstacles=False):
        # Devolve os vizinhos da célula (4 ou 8 conforme a configuração).
        x, y = vertex
        if self.exploration_setting == '4N':
            movements = get_movements_4n(x=x, y=y)
        else:
            movements = get_movements_8n(x=x, y=y)
        # (apenas estético: alterna a ordem para os caminhos ficarem mais "direitos")
        if (x + y) % 2 == 0:
            movements.reverse()
        return list(self.filter(neighbors=movements, avoid_obstacles=avoid_obstacles))

    def set_obstacle(self, pos):
        # Marca a célula como parede (255).
        row, col = round(pos[0]), round(pos[1])
        self.occupancy_grid_map[row, col] = OBSTACLE

    def remove_obstacle(self, pos):
        # Marca a célula como livre (0).
        row, col = round(pos[0]), round(pos[1])
        self.occupancy_grid_map[row, col] = UNOCCUPIED
