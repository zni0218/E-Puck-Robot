"""
utils.py  --  PEÇAS DE APOIO DO PLANEADOR
=========================================
A heurística (distância em linha reta) e as listas de movimentos (vizinhos).
"""

import math
from typing import List


class Vertex:
    # (Usado pela versão com replaneamento; mantido por compatibilidade.)
    def __init__(self, pos):
        self.pos = pos
        self.edges_and_costs = {}

    def add_edge_with_cost(self, succ, cost):
        if succ != self.pos:
            self.edges_and_costs[succ] = cost

    @property
    def edges_and_c_old(self):
        return self.edges_and_costs


class Vertices:
    def __init__(self):
        self.list = []

    def add_vertex(self, v):
        self.list.append(v)

    @property
    def vertices(self):
        return self.list


def heuristic(p, q):
    # Distância EUCLIDIANA (linha reta) entre duas células. É a heurística correta para
    # grelha 8-conexa (admissível: nunca sobre-estima, porque a diagonal vale ~1.414).
    # (Manhattan sobre-estimaria a diagonal -> caminhos sub-ótimos. Por isso, euclidiana.)
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def get_movements_4n(x, y) -> List:
    # 4 vizinhos: cima, baixo, esquerda, direita (sem diagonais).
    return [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]


def get_movements_8n(x, y) -> List:
    # 8 vizinhos: os 4 anteriores + as 4 diagonais.
    return [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1),
            (x + 1, y + 1), (x - 1, y + 1), (x - 1, y - 1), (x + 1, y - 1)]
