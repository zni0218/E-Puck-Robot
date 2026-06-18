"""
parking_map.py  --  A PONTE ENTRE A IMAGEM E A GRELHA DO PLANEADOR
==================================================================
Responsabilidade ÚNICA: pegar no PNG do mapa e produzir uma grelha de células
onde cada célula é "livre" ou "ocupada", mais as funções para converter entre
metros (mundo do Webots) e células (onde o D* Lite trabalha).

NÃO faz planeamento (isso é o d_star_lite.py) e NÃO descobre os lugares de
estacionamento (isso é o parking_other_epucks.py). Mantém-se pequena de propósito.

OS TRÊS REFERENCIAIS (é aqui que toda a gente se baralha):
  - PIXEL da imagem (linha, coluna): origem no canto superior-esquerdo, linha cresce
    para BAIXO. É como o PNG é lido.
  - MUNDO do Webots (x, y) em metros: x cresce para a direita, y cresce para CIMA.
    É o que o GPS devolve. Relação: x = res*coluna ; y = oy - res*linha.
  - CÉLULA da grelha (gx, gy): gx = linha, gy = coluna, em células de 'cell' metros.

SOBRE cell_size (a tua pergunta):
  cell_size É o comprimento da aresta entre dois nós (vértices) vizinhos. O teu modelo
  mental (vértices ligados por arestas) está CERTO. Mas atenção: a grelha representa
  paredes como CÉLULAS OCUPADAS. Se uma parede for mais fina que uma célula e "couber"
  entre dois centros de células livres, o planeador NÃO a vê e corta por ela.
  As divisórias entre lugares têm só ~2.4 cm. Por isso a célula tem de ser pequena
  (2 cm) — com 18.4 cm (espaçamento entre lugares) as divisórias misturavam-se com o
  espaço livre e, como o e-puck (7.4 cm) é menor que uma célula, "livre" não garantiria
  que ele lá passa. Daí: grelha fina (2 cm) + inflação das paredes.
"""

import math
import numpy as np
import yaml
from PIL import Image

# A inflação (engordar paredes) usa scipy se existir; senão há um fallback manual.
try:
    from scipy.ndimage import binary_dilation
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class ParkingMap:
    def __init__(self, png_path, yaml_path, cell_size=0.02, inflate_m=0.05):
        # cell_size : comprimento da aresta da grelha, em metros (2 cm = bom compromisso).
        # inflate_m : quanto engordar as paredes (≈ raio do e-puck + margem). É GEOMETRIA,
        #             não nada aprendido: garante que o robô nunca raspa as paredes.

        # 1) Ler o YAML: dá a resolução (metros por pixel) e o limiar de "parede".
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.res = float(cfg["resolution"])               # metros por pixel (0.008)
        self.occ_thresh = float(cfg["occupied_thresh"])   # 0..1; abaixo disto = parede

        # 2) Abrir o PNG em tons de cinza e marcar como parede os pixéis escuros.
        img = Image.open(png_path).convert("L")
        self.px = np.array(img)                           # matriz (linhas, colunas)
        self.h_px, self.w_px = self.px.shape              # altura, largura em pixéis
        self.wall_px = self.px <= int(255 * self.occ_thresh)  # True onde é parede

        # 3) Origem do mundo: o canto superior-esquerdo do PNG é (x=0, y=oy).
        #    oy = altura do mapa em metros (o topo do mapa). y cresce para cima.
        self.ox = 0.0
        self.oy = self.h_px * self.res                    # ex.: 255 * 0.008 = 2.04 m

        # 4) Tamanho da grelha do planeador (nº de linhas gx e colunas gy).
        self.cell = cell_size
        self.gx = int(math.ceil(self.h_px * self.res / self.cell))   # linhas da grelha
        self.gy = int(math.ceil(self.w_px * self.res / self.cell))   # colunas da grelha

        # 5) Construir a ocupação (que células são parede) e engordar as paredes.
        self.occ_raw = self._build_occupancy()            # paredes "cruas"
        self.occ = self._inflate(self.occ_raw, inflate_m) # paredes engordadas (a que se usa)

    # ----------------------- CONVERSÕES mundo <-> célula ----------------------- #
    def world_to_cell(self, x, y):
        # Metros (x,y) -> índices de célula (gx, gy). y invertido porque a linha 0
        # da grelha é o TOPO do mapa (y máximo).
        gx = int((self.oy - y) / self.cell)   # linha  (quanto mais alto y, menor a linha)
        gy = int((x - self.ox) / self.cell)   # coluna (x cresce com a coluna)
        return (gx, gy)

    def cell_to_world(self, gx, gy):
        # Índices de célula -> metros do CENTRO da célula (+0.5 = meio da célula).
        y = self.oy - (gx + 0.5) * self.cell
        x = self.ox + (gy + 0.5) * self.cell
        return (x, y)

    def in_bounds(self, cell):
        # A célula está dentro dos limites da grelha?
        gx, gy = cell
        return 0 <= gx < self.gx and 0 <= gy < self.gy

    def is_free_cell(self, cell):
        # A célula está dentro dos limites E não é parede (na grelha já engordada)?
        gx, gy = cell
        return self.in_bounds(cell) and not self.occ[gx, gy]

    # ----------------------- CONSTRUÇÃO da grelha ----------------------- #
    def _build_occupancy(self):
        # Constrói a matriz booleana de ocupação (True = parede) a partir do PNG.
        occ = np.zeros((self.gx, self.gy), dtype=bool)    # tudo livre de início
        rows, cols = np.nonzero(self.wall_px)             # pixéis que são parede
        for r, c in zip(rows, cols):                      # para cada pixel-parede:
            wx = self.ox + c * self.res                   #   coluna do pixel -> x no mundo
            wy = self.oy - r * self.res                   #   linha  do pixel -> y no mundo
            gx, gy = self.world_to_cell(wx, wy)           #   mundo -> célula da grelha
            if 0 <= gx < self.gx and 0 <= gy < self.gy:   #   dentro da grelha?
                occ[gx, gy] = True                        #   marca a célula como parede
        return occ                                        # True = parede, False = livre

    def _inflate(self, occ, inflate_m):
        # "Engorda" as paredes inflate_m metros, dilatando a ocupação. Isto cria a
        # margem de segurança: o planeador trata as paredes como mais largas, por isso
        # o caminho nunca passa colado a elas (o robô tem raio ~3.7 cm).
        iters = max(0, int(round(inflate_m / self.cell)))  # quantas células engordar
        if iters == 0:
            return occ.copy()
        if _HAS_SCIPY:
            return binary_dilation(occ, iterations=iters)  # caminho rápido (scipy)
        # Fallback sem scipy: propaga a ocupação aos 4 vizinhos, 'iters' vezes.
        out = occ.copy()
        for _ in range(iters):
            nxt = out.copy()
            nxt[1:, :]  |= out[:-1, :]
            nxt[:-1, :] |= out[1:, :]
            nxt[:, 1:]  |= out[:, :-1]
            nxt[:, :-1] |= out[:, 1:]
            out = nxt
        return out


    # ----------------------- GEOMETRIA DO LUGAR (para entrar a direito) ----------------------- #
    def raw_wall_at(self, x, y):
        # Há parede no ponto (x,y)? Usa o mapa CRU (sem inflação) — para medir o lugar real.
        r = int(round((self.oy - y) / self.res))
        c = int(round((x - self.ox) / self.res))
        if r < 0 or c < 0 or r >= self.h_px or c >= self.w_px:
            return True
        return bool(self.wall_px[r, c])

    def slot_geometry(self, x, y, max_m=0.6, step=0.004):
        # A partir do centro do lugar (x,y), descobre:
        #   - inward: vetor unitário (dx,dy) que aponta PARA DENTRO (para a parede do fundo)
        #   - back:   distância (m) do centro até essa parede do fundo
        # COMO: mede a distância à parede nas 4 direções; a SAÍDA do lugar é a direção MAIS
        # ABERTA (dá para o hall), e "para dentro" é o oposto da saída. Isto funciona seja o
        # lugar virado para Norte, Sul, Este ou Oeste -> generaliza para outros mapas.
        dist = {}
        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            d = step
            while d < max_m:
                if self.raw_wall_at(x + dx * d, y + dy * d):
                    break
                d += step
            dist[(dx, dy)] = d
        opening = max(dist, key=dist.get)          # direção mais aberta = saída do lugar
        inward = (-opening[0], -opening[1])         # para dentro = oposto da saída
        return inward, dist[inward]

if __name__ == "__main__":
    # Teste rápido: imprime o tamanho do mapa e da grelha.
    import os
    here = "C:/Users/user/Documents/Webots/Worlds/IRI_public_TP_classes-master/IRI_public_TP_classes-master/controllers/Projeto_Dlite/worlds"
    pm = ParkingMap(os.path.join(here, "Scenario1.png"),
                    os.path.join(here, "Scenario1_config.yaml"))
    print(f"Mapa: {pm.w_px}x{pm.h_px}px -> {pm.w_px*pm.res:.3f}x{pm.h_px*pm.res:.3f} m")
    print(f"Grelha: {pm.gx}x{pm.gy} células de {pm.cell*100:.0f} cm")
    print(f"Células-parede (cruas): {pm.occ_raw.sum()}  ->  (infladas): {pm.occ.sum()}")