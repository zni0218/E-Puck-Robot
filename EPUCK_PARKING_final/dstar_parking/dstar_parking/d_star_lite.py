"""
d_star_lite.py  --  O ALGORITMO DE PLANEAMENTO
==============================================

D-Star Lite é um algoritmo de path planning incremental:
ele recalcula caminhos de forma eficiente quando o mapa muda,
em vez de recalcular tudo do zero.

Ideia principal:
- g[s]   → custo conhecido atual de s até ao objetivo
- rhs[s] → melhor estimativa "one-step lookahead"

Quando g[s] == rhs[s], o nó está consistente.
Quando não são iguais, o nó precisa de ser corrigido.

A fila U guarda os nós inconsistentes e processa primeiro os mais importantes.
"""

from priority_queue import PriorityQueue, Priority
from grid import OccupancyGridMap
import numpy as np
from utils import heuristic

OBSTACLE = 255
UNOCCUPIED = 0


class DStarLite:

    def __init__(self, map: OccupancyGridMap, s_start, s_goal):

        # guarda início, objetivo e estado atual do algoritmo
        self.s_start = s_start
        self.s_goal = s_goal
        self.s_last = s_start

        # termo usado para ajustes quando o início muda (replanning dinâmico)
        self.k_m = 0

        # fila de prioridade com os nós "por corrigir"
        self.U = PriorityQueue()

        # =====================================================
        # TABELAS PRINCIPAIS DO ALGORITMO
        # =====================================================

        # g = custo conhecido (inicialmente infinito)
        self.rhs = np.ones((map.x_dim, map.y_dim)) * np.inf
        self.g = self.rhs.copy()

        # grelha onde o algoritmo "vê" o mundo (paredes, espaço livre)
        self.sensed_map = OccupancyGridMap(
            x_dim=map.x_dim,
            y_dim=map.y_dim,
            exploration_setting='8N'
        )

        # objetivo: custo até si próprio é zero
        self.rhs[self.s_goal] = 0

        # insere o objetivo na fila como ponto inicial de propagação
        self.U.insert(
            self.s_goal,
            Priority(
                heuristic(self.s_start, self.s_goal),
                0
            )
        )

    # =========================================================
    # FUNÇÃO DE PRIORIDADE
    # =========================================================
    def calculate_key(self, s):

        # calcula prioridade de um nó na fila U
        # menor chave = processado primeiro

        k1 = min(self.g[s], self.rhs[s]) + heuristic(self.s_start, s) + self.k_m
        k2 = min(self.g[s], self.rhs[s])

        return Priority(k1, k2)

    # =========================================================
    # CUSTO ENTRE DOIS NÓS
    # =========================================================
    def c(self, u, v):

        # define custo de movimento entre células

        # se qualquer célula for obstáculo → custo infinito
        if not self.sensed_map.is_unoccupied(u) or not self.sensed_map.is_unoccupied(v):
            return float('inf')

        # caso contrário, usa distância euclidiana (1 ou ~1.4 em diagonais)
        return heuristic(u, v)

    # =========================================================
    # VERIFICAÇÃO NA FILA
    # =========================================================
    def contain(self, u):

        # verifica se o nó está na fila de prioridade
        return u in self.U.vertices_in_heap

    # =========================================================
    # ATUALIZAÇÃO DE UM VÉRTICE
    # =========================================================
    def update_vertex(self, u):

        # mantém consistência entre g e rhs

        if self.g[u] != self.rhs[u] and self.contain(u):
            # nó ainda inconsistente e já na fila → apenas atualiza prioridade
            self.U.update(u, self.calculate_key(u))

        elif self.g[u] != self.rhs[u] and not self.contain(u):
            # inconsistente mas não está na fila → insere
            self.U.insert(u, self.calculate_key(u))

        elif self.g[u] == self.rhs[u] and self.contain(u):
            # já consistente → remove da fila
            self.U.remove(u)

    # =========================================================
    # CÁLCULO DO CAMINHO MAIS CURTO
    # =========================================================
    def compute_shortest_path(self):

        # continua a processar enquanto houver inconsistências relevantes
        while self.U.top_key() < self.calculate_key(self.s_start) \
                or self.rhs[self.s_start] != self.g[self.s_start]:

            u = self.U.top()          # nó com maior prioridade
            k_old = self.U.top_key()  # chave atual na fila
            k_new = self.calculate_key(u)  # chave recalculada

            # caso a prioridade esteja desatualizada
            if k_old < k_new:
                self.U.update(u, k_new)

            # =================================================
            # CASO 1: g é demasiado grande (precisa de diminuir)
            # =================================================
            elif self.g[u] > self.rhs[u]:

                # corrige g para o melhor valor conhecido
                self.g[u] = self.rhs[u]
                self.U.remove(u)

                # propaga atualização para os vizinhos
                for s in self.sensed_map.succ(vertex=u):
                    if s != self.s_goal:
                        self.rhs[s] = min(
                            self.rhs[s],
                            self.c(s, u) + self.g[u]
                        )
                    self.update_vertex(s)

            # =================================================
            # CASO 2: g ficou demasiado pequeno (inconsistência)
            # =================================================
            else:

                self.g_old = self.g[u]
                self.g[u] = float('inf')

                pred = self.sensed_map.succ(vertex=u)
                pred.append(u)

                for s in pred:

                    if self.rhs[s] == self.c(s, u) + self.g_old:

                        if s != self.s_goal:

                            min_s = float('inf')

                            for s_ in self.sensed_map.succ(vertex=s):

                                temp = self.c(s, s_) + self.g[s_]

                                if min_s > temp:
                                    min_s = temp

                            self.rhs[s] = min_s

                    # reavalia consistência do nó
                    self.update_vertex(u)