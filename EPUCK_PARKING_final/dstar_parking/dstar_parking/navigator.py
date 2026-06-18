"""
navigator.py  --  EMBRULHO SIMPLES À VOLTA DO D* Lite
=====================================================
O d_star_lite.py é genérico e um pouco difícil de usar diretamente. Esta classe
esconde-o atrás de 3 ideias:
  1) criar  -> dá-se o mapa, o início e o fim; ele planeia logo.
  2) caminho -> devolve a lista de células do início ao fim.
  3) waypoints -> a mesma lista, mas já convertida para metros (para o robô seguir).

PORQUÊ "mapa conhecido" (e não o modo cego do D* Lite):
  Temos o mapa todo a partir do PNG. Não precisamos de descobrir paredes com o Lidar.
  Damos o mapa inteiro ao D* Lite de uma vez e ele planeia o caminho completo. (O D* Lite
  brilha quando o mapa é desconhecido e muda — não é o nosso caso, mas é o algoritmo pedido.)

DUAS GRELHAS, NÃO TE BARALHES:
  - ParkingMap  -> a ponte com o mundo real (PNG, metros, células, inflação). É a 'pm'.
  - OccupancyGridMap (grid.py) -> a estrutura de dados que o ALGORITMO usa por dentro
    (quem são os vizinhos? a célula é livre?). É a 'base' aqui em baixo.
  Esta classe copia a ocupação da ParkingMap para a OccupancyGridMap e arranca o D* Lite.
"""

import numpy as np

from grid import OccupancyGridMap, OBSTACLE
from d_star_lite import DStarLite


class DStarNavigator:
    def __init__(self, pmap, start_cell, goal_cell):
        # 1) Guardar a ponte com o mundo e as dimensões da grelha.
        self.pm = pmap
        self.goal_cell = goal_cell

        # 2) Criar a grelha interna do algoritmo (vazia) e o objeto D* Lite.
        base = OccupancyGridMap(pmap.gx, pmap.gy, exploration_setting="8N")
        self.dstar = DStarLite(map=base, s_start=start_cell, s_goal=goal_cell)

        # 3) Copiar TODAS as paredes da ParkingMap para a grelha do algoritmo.
        #    (np.nonzero(pmap.occ) dá as células-parede; marcamo-las como obstáculo.)
        sensed = self.dstar.sensed_map
        for gx, gy in zip(*np.nonzero(pmap.occ)):
            sensed.occupancy_grid_map[int(gx), int(gy)] = OBSTACLE

        # 4) Planear o caminho mais curto start -> goal (uma só vez).
        self.dstar.compute_shortest_path()

    def full_path(self, max_len=4000):
        # Reconstrói o caminho seguindo, a cada passo, o vizinho de menor custo-até-ao-goal.
        # (O D* Lite guarda em g[célula] o custo de cada célula até ao objetivo.)
        path = [self.dstar.s_start]
        cur = self.dstar.s_start
        seen = {cur}
        while cur != self.goal_cell and len(path) < max_len:
            best, arg = float("inf"), None
            for nxt in self.dstar.sensed_map.succ(cur, avoid_obstacles=False):
                custo = self.dstar.c(cur, nxt) + self.dstar.g[nxt]   # passo + resto até ao goal
                if custo < best:
                    best, arg = custo, nxt
            # se não há para onde ir, ou repetimos, ou o custo é infinito -> parar
            if arg is None or arg in seen or self.dstar.g[arg] == float("inf"):
                break
            path.append(arg); seen.add(arg); cur = arg
        return path                                  # lista de células (gx, gy)

    def world_waypoints(self):
        # O mesmo caminho, mas cada célula convertida para o seu centro em metros.
        return [self.pm.cell_to_world(gx, gy) for (gx, gy) in self.full_path()]
