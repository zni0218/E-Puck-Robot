"""
parking_other_epucks.py  --  O CENÁRIO: coloca os carros e escolhe o lugar livre
================================================================================
Antes de o robô andar, isto coloca os 7 carros nos lugares e marca 1 como vazio,
devolvendo as coordenadas (x, y, z) desse lugar livre -> é o GOAL do D* Lite.

É a ÚNICA fonte da geometria dos lugares (a ParkingMap já NÃO os calcula — antes
eram calculados duas vezes, o que era redundante; ficou só aqui).

Como acha os lugares: para cada pixel livre mede a distância às paredes nas 4
direcções; um lugar é um sítio com paredes em pelo menos 3 lados. Agrupa esses
pixéis em componentes (cada componente = 1 lugar) e calcula o ponto de spawn a
1/3 da profundidade a partir da abertura, centrado entre as paredes laterais.
"""
import yaml
import numpy as np
import random
import os
from PIL import Image
from controller import Supervisor
from collections import deque


def parked_other_cars(supervisor=None):
    # ------------------------------------------------------------------- #
    # Caminho relativo ao próprio ficheiro — funciona em qualquer máquina #
    # ------------------------------------------------------------------- #

    base_dir = os.path.dirname(__file__)
    path_to_worlds = os.path.join(base_dir, "worlds")

    yaml_filepath = os.path.join(path_to_worlds, "Scenario1_config.yaml")
    with open(yaml_filepath, 'r') as stream:
        yaml_data = yaml.safe_load(stream)

    image_filename       = yaml_data['image']
    resolution           = yaml_data['resolution']
    occupied_thresh      = yaml_data['occupied_thresh']
    max_pixel_value_wall = int(255 * occupied_thresh)

    img    = Image.open(os.path.join(path_to_worlds, image_filename)).convert('L')
    np_img = np.array(img)

    height, width = np_img.shape

    grid = np.zeros((height, width), dtype=np.uint8)
    grid[np_img <= max_pixel_value_wall] = 1   # 1 = parede

    # ------------------------------------------------------------------ #
    # Distâncias às paredes nas 4 direcções                              #
    # ------------------------------------------------------------------ #
    OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def dist_to_wall(grid, y, x, max_dist):
        h, w = grid.shape
        d = {'N': max_dist + 1, 'S': max_dist + 1,
             'E': max_dist + 1, 'W': max_dist + 1}
        for i in range(1, max_dist + 1):
            if y - i < 0:    break
            if grid[y - i][x] == 1: d['N'] = i; break
        for i in range(1, max_dist + 1):
            if y + i >= h:   break
            if grid[y + i][x] == 1: d['S'] = i; break
        for i in range(1, max_dist + 1):
            if x + i >= w:   break
            if grid[y][x + i] == 1: d['E'] = i; break
        for i in range(1, max_dist + 1):
            if x - i < 0:   break
            if grid[y][x - i] == 1: d['W'] = i; break
        return d

    # ------------------------------------------------------------------ #
    # Candidatos: pixeis livres com paredes em pelo menos 3 direcções    #
    # ------------------------------------------------------------------ #
    max_dist_cells     = 11
    possible_spawn_pixels = []
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0:
                d = dist_to_wall(grid, y, x, max_dist_cells)
                close_dirs = sum(1 for v in d.values() if v <= max_dist_cells)
                if close_dirs >= 3:
                    possible_spawn_pixels.append((y, x))

    # ------------------------------------------------------------------ #
    # Componentes conexas → um candidato por componente                  #
    # ------------------------------------------------------------------ #
    def find_components(pixels):
        pixels = set(pixels)
        components = []
        while pixels:
            start = pixels.pop()
            queue = deque([start])
            comp  = [start]
            while queue:
                y, x = queue.popleft()
                for ny, nx in [(y+1,x),(y-1,x),(y,x+1),(y,x-1)]:
                    if (ny, nx) in pixels:
                        pixels.remove((ny, nx))
                        queue.append((ny, nx))
                        comp.append((ny, nx))
            components.append(comp)
        return components

    components = find_components(possible_spawn_pixels)

    # ------------------------------------------------------------------- #
    # Para cada componente:                                               #
    #   1. Pixel representativo = centroide (em pixeis)                   #
    #   2. Medir distâncias a partir desse pixel                          #
    #   3. Abertura = direcção com MAX distância (não bateu na parede)    #
    #   4. Laterais = as 2 direcções perpendiculares à abertura           #
    #   5. Centro lateral: deslocar para equalizar as distâncias laterais #
    #   6. Posição final: 1/3 da profundidade total a partir da abertura  #
    # ------------------------------------------------------------------- #
    PERP = {
        'N': ('E', 'W'), 'S': ('E', 'W'),
        'E': ('N', 'S'), 'W': ('N', 'S'),
    }
    # Deslocamento em (dy, dx) para cada direcção
    DELTA = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}

    spawn_pixels = []   # lista de (row_float, col_float)

    for comp in components:
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        cy = int(round(sum(ys) / len(ys)))
        cx = int(round(sum(xs) / len(xs)))

        d = dist_to_wall(grid, cy, cx, max_dist_cells)

        # 3. Direcção de abertura (a que ficou sem parede dentro do alcance)
        open_dir   = max(d, key=d.get)
        closed_dir = OPPOSITE[open_dir]

        d_open   = d[open_dir]    # e.g. 16 (abertura → Sul)
        d_closed = d[closed_dir]  # e.g.  6 (fundo   → Norte)

        # 4 & 5. Centrar lateralmente
        lat1, lat2 = PERP[open_dir]           # e.g. 'E' e 'W'
        d_lat1, d_lat2 = d[lat1], d[lat2]
        # Deslocar metade da diferença na direcção com mais espaço
        lateral_shift = (d_lat1 - d_lat2) // 2
        dy_lat, dx_lat = DELTA[lat1]
        cy_center = cy + lateral_shift * dy_lat
        cx_center = cx + lateral_shift * dx_lat

        # 6. Posição de spawn: 1/3 da profundidade total a partir da borda da abertura
        #    borda da abertura em pixeis:
        total_depth = d_open + d_closed
        dy_open, dx_open = DELTA[open_dir]
        # borda da abertura (pixel no limite entre slot e corredor):
        row_borda = cy_center + d_open * dy_open
        col_borda = cx_center + d_open * dx_open
        # recuar 1/3 do total para dentro (direcção do fundo):
        dy_closed, dx_closed = DELTA[closed_dir]
        third = total_depth // 3
        row_spawn = row_borda + third * dy_closed   # dy_closed é o oposto de dy_open
        col_spawn = col_borda + third * dx_closed

        spawn_pixels.append((row_spawn, col_spawn))

    spawn_pixels = sorted(spawn_pixels, key=lambda c: c[0])
    print("Spawn pixels (row, col):", spawn_pixels)

    # ------------------------------------------------------------------ #
    # Converter pixeis → coordenadas mundo                               #
    # origin: canto superior esquerdo = (ox=0.0, oy=2.04)                #
    # wx = ox + resolution * col                                         #
    # wy = oy - resolution * row                                         #
    # ------------------------------------------------------------------ #

    img_world_shape = (height * resolution, width * resolution)

    ox, oy = 0.0, 2.04
    spawn_coords = []
    for (ry, cx) in spawn_pixels:
        wx = ox + resolution * cx
        wy = oy - resolution * ry
        spawn_coords.append((wx, wy, 0.0))

    # Ordenar a lista, assim o primeiro é o mais próximo de canto superior esquerdo
    spawn_coords = sorted(spawn_coords, key=lambda c: c[0])

    # Trunca 3 casas decimais
    def trunc3(x):
        return int(x * 1000) / 1000

    spawn_coords = [(trunc3(x), trunc3(y), trunc3(z)) for x, y, z in spawn_coords]

    print("Spawn coords (x, y, z):", spawn_coords)
    print(f"Encontrados {len(spawn_coords)} lugares de estacionamento.")
    print(f"Vais spawnar {len(spawn_coords)-1} e-pucks.")

    # ------------------------------------------------------------------ #
    # Supervisor: posicionar os e-pucks                                  #
    # ------------------------------------------------------------------ #
    if supervisor is None:
        supervisor = Supervisor()
    free_spot = random.randint(0, len(spawn_coords) - 1)
    #free_spot = 7   # índice 1-based; lugar 7 fica livre
    goal = None

    n = 1
    for i, (x, y, z) in enumerate(spawn_coords):
        if i + 1 != free_spot:
            robot = supervisor.getFromDef(f"EPUCK{n}")
            if robot is None:
                print(f"ERRO: Não existe DEF EPUCK{n} no .wbt")
                n += 1
                continue
            robot.getField("translation").setSFVec3f([x, y, z])
            robot.getField("rotation").setSFRotation([0, 0, 1, 0])
            print(f"EPUCK{n} colocado em ({x:.3f}, {y:.3f}, {z:.3f})")
            n += 1
        else:
            goal = (x, y, z)
            print(f"Lugar livre (goal): ({x:.3f}, {y:.3f}, {z:.3f})")

    return img_world_shape, goal, spawn_coords


if __name__ == '__main__':
    img_shape, goal, parking_slots = parked_other_cars()
    xs = [p[0] for p in parking_slots]
    dx = np.diff(xs) 
    print(img_shape[0], img_shape[1])
    print("mean dx:", dx.mean())
    print("std dx:", dx.std())
    print(f"\nGoal: {goal}")


